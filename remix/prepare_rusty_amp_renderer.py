#!/usr/bin/env python3
"""Build the pinned Apache-2.0 rusty-amp no-cabinet stage renderer."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


PINNED_COMMIT = "831d6bba4ad3a1a8da958c510cbc96a9e55b73f9"
ORIGINAL_MARSHALL_SHA256 = "c23157baba879423770d12ad4617cc1f7799c96c0d63bace725e4c6ff34cd2a1"
STRUCT_INSERT = """

/// Observable no-cabinet boundaries for deterministic offline supervision.
#[derive(Clone, Copy, Debug)]
pub struct MarshallStageTaps {
    pub front_end: f32,
    pub preamp: f32,
    pub tone_stack: f32,
    pub voice_balance: f32,
    pub power_amp: f32,
    pub output_transformer: f32,
    pub speaker_load: f32,
    pub output: f32,
}
"""
TAP_METHOD = """

    /// Process one sample while exposing each base-rate stage boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn process_taps(
        &mut self,
        sample: f32,
        gain: f32,
        bass: f32,
        mid: f32,
        treble: f32,
        presence: f32,
        master: f32,
    ) -> MarshallStageTaps {
        if self.tone_cache.changed(bass, mid, treble) {
            self.update_tone_stack(bass, mid, treble);
        }
        if self.presence_cache.changed(presence) {
            self.update_presence(presence);
        }
        let front_end = self.bright.process(self.front.process(sample), gain);
        let pregain = 1.0 + gain * 39.0;
        let bias = self.bloom.follow(front_end) * 0.06;
        let g1 = pregain.powf(0.6) * 1.4;
        let g2 = (pregain / pregain.powf(0.6)) * 1.6;
        let up = self.os.upsample(front_end);
        let mut down = [0.0f32; 8];
        for (o, &u) in down.iter_mut().zip(up.iter()) {
            let u = self.pre_clip_hp.process(u);
            let d = self.grid.shift(self.cathode.shift((u + bias) * g1));
            let s = tube_clip_asym(d) / g1.sqrt();
            let s = self.stage_hp.process(s);
            *o = tube_clip_asym(s * g2) / g2.sqrt();
        }
        let preamp = self.os.downsample(down);
        let tone_stack = self.tone.process(preamp);
        let voice_balance = self.voice.process(tone_stack);
        let power_amp = self.power_amp(voice_balance);
        let output_transformer = self.xfmr.process(power_amp);
        let speaker_load = self.speaker.process(output_transformer, self.envelope);
        let output = self
            .out_hp
            .process(self.presence.process(speaker_load, self.envelope))
            * master
            * 6.0;
        MarshallStageTaps {
            front_end,
            preamp,
            tone_stack,
            voice_balance,
            power_amp,
            output_transformer,
            speaker_load,
            output,
        }
    }
"""
TRAIT_IMPL = """impl Amplifier for Marshall {
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn process(
        &mut self,
        sample: f32,
        gain: f32,
        bass: f32,
        mid: f32,
        treble: f32,
        presence: f32,
        master: f32,
    ) -> f32 {
        self.process_taps(sample, gain, bass, mid, treble, presence, master)
            .output
    }
}

"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(workspace: Path, source: Path) -> Path:
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"unexpected rusty-amp commit: {commit}")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise ValueError("rusty-amp source must be clean before instrumentation")
    marshall = source / "src/dsp/amp/marshall.rs"
    if _sha256(marshall) != ORIGINAL_MARSHALL_SHA256:
        raise ValueError("pinned rusty-amp Marshall source hash changed")
    text = marshall.read_text()
    text = text.replace("\nimpl Marshall {", STRUCT_INSERT + "\nimpl Marshall {", 1)
    trait_start = text.index("impl Amplifier for Marshall {")
    next_item = text.index("/// Asymmetric 12AX7 triode waveshaper.", trait_start)
    text = text[:trait_start].rstrip()[:-1] + TAP_METHOD + "\n}\n\n" + TRAIT_IMPL + text[next_item:]
    marshall.write_text(text)
    renderer = source / "examples/muspector_stage_render.rs"
    shutil.copyfile(workspace / "remix/rusty_amp_stage_render.rs", renderer)
    subprocess.run(
        ["cargo", "test", "--lib", "--no-default-features", "-q"],
        cwd=source,
        check=True,
    )
    subprocess.run(
        ["cargo", "build", "--release", "--example", "muspector_stage_render"],
        cwd=source,
        check=True,
    )
    return source / "target/release/examples/muspector_stage_render"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    binary = prepare(args.workspace.resolve(), args.source.resolve())
    print(f"{binary}\nsha256={_sha256(binary)}")


if __name__ == "__main__":
    main()

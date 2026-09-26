use rusty_amp::dsp::amp::Marshall;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn read_mono(path: &Path) -> (u32, Vec<f32>) {
    let mut reader = hound::WavReader::open(path).expect("open input WAV");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "renderer requires mono input");
    let values = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|v| v.expect("read float sample")).collect(),
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader.samples::<i16>().map(|v| v.expect("read i16 sample") as f32 / 32768.0).collect(),
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(spec.bits_per_sample as i32 - 1);
            reader.samples::<i32>().map(|v| v.expect("read i32 sample") as f32 / scale).collect()
        }
    };
    (spec.sample_rate, values)
}

fn write_float(path: PathBuf, rate: u32, values: &[f32]) {
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut writer = hound::WavWriter::create(path, spec).expect("create output WAV");
    for &value in values { writer.write_sample(value).expect("write sample"); }
    writer.finalize().expect("finalize output WAV");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 9, "input output-dir gain bass mid treble presence master");
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    let controls: Vec<f32> = args[3..].iter().map(|v| v.parse::<f32>().expect("numeric control")).collect();
    assert!(controls.iter().all(|v| (0.0..=1.0).contains(v)));
    fs::create_dir_all(output).expect("create output directory");
    let (rate, clean) = read_mono(input);
    let mut amp = Marshall::new(rate as f32);
    let mut stages = vec![Vec::with_capacity(clean.len()); 8];
    for sample in clean.iter().copied() {
        let taps = amp.process_taps(sample, controls[0], controls[1], controls[2], controls[3], controls[4], controls[5]);
        for (target, value) in stages.iter_mut().zip([taps.front_end, taps.preamp, taps.tone_stack, taps.voice_balance, taps.power_amp, taps.output_transformer, taps.speaker_load, taps.output]) { target.push(value); }
    }
    for (index, values) in stages.iter().enumerate() { write_float(output.join(format!("stage-{}.wav", index + 1)), rate, values); }
}

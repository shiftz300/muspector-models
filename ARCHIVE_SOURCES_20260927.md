# Local training-resource provenance — 2026-09-27

This index records where the soon-to-be-deleted local corpus came from. The
authoritative use-specific rights and integrity records are
`remix/data_sources.json`, `train/sources.json`, and `train/LICENSES.md` in this
repository. Recheck the publisher's current terms before any future download,
training, model redistribution, or audio redistribution. A source URL is a
locator, **not** permission to reuse its content for every purpose.

The local dataset was `../muspector/data/`; `data` in this model checkout was
only a symlink. Its top-level `corpus/` names map as follows:

| Local corpus directory | Source record / recovery locator | Boundary |
| --- | --- | --- |
| `aachen-chapel-rir` | [Zenodo 20428705](https://zenodo.org/records/20428705) | See registry |
| `apple-au` | Local renders from macOS AUDistortion, AUDelay and AUMatrixReverb over GuitarJam/GuitarSet Clean sources | **Not a downloadable corpus**; exact local renders and their ignored manifest will be lost |
| `asrnn-official-results`, `asrnn-physical-effects` | [Zenodo 20406285](https://zenodo.org/records/20406285) | Noncommercial research boundary |
| `but-reverbdb` | [Official BUT ReverbDB page](https://speech.fit.vut.cz/software/but-speech-fit-reverb-database) | Check original room/audio terms |
| `eg-ipt`, `eg-ipt-sm57` | [Zenodo 15205644](https://zenodo.org/records/15205644) | Fixed physical amp/cab/mic pairing, not generic Amp validation |
| `egdb-pg-subset-v1` | [Zenodo 19789500](https://zenodo.org/records/19789500) | Partial local extraction; range/audit method in source code |
| `egfxset` | [Zenodo 7044411](https://zenodo.org/records/7044411) | Recognition domain, not automatic restoration ground truth |
| `freepats-electric-guitar-direct` | [FreePats electric guitar](https://freepats.zenvoid.org/ElectricGuitar/clean-electric-guitar.html) | Clean note source |
| `guitar-effects-chains` | [Zenodo 7871720](https://zenodo.org/records/7871720) | DAFx/software chains; order/presence only for archived Wet |
| `guitar-techs` | [Zenodo 14963133](https://zenodo.org/records/14963133) | P1/P2 Clean only in the active product contract |
| `guitarjam` | [Hugging Face GuitarJam](https://huggingface.co/datasets/Julian-br/GuitarJam) | Pin revision in `train/sources.json` |
| `guitarset` | [Zenodo 3371780](https://zenodo.org/records/3371780) | Original archives/checksums in `train/sources.json` |
| `idmt-smt-audio-effects` | [Fraunhofer IDMT page](https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html) | Noncommercial/no-derivatives; excluded from product gradients |
| `karoryfer-emilyguitar` | [Karoryfer Emilyguitar](https://github.com/sfzinstruments/karoryfer.emilyguitar) | Clean note source |
| `longitudinal-guitar-string-ageing` | [Zenodo 19823590](https://zenodo.org/records/19823590) | Repeated routine; not independent player diversity |
| `marshall-jvm410h` | [Zenodo 7970723](https://zenodo.org/records/7970723) | Single-device diagnostic |
| `multimodal-electric-guitar` | [Zenodo 6470236](https://zenodo.org/records/6470236) | Positive-only presence evidence |
| `ok5-rir` | [Zenodo 18622201](https://zenodo.org/records/18622201) | External room evidence |
| `openair-external-v1` | [OpenAIR project](https://www.york.ac.uk/physics-engineering-technology/research/communication-technologies/projects/open-acoustic-impulse-response-library/) | External room evidence |
| `openslr26-simulated-rir` | [OpenSLR 26](https://www.openslr.org/26) | Simulated rooms, not physical-room proof |
| `remfx-1-1` | [RemFX official evaluation pairs](https://zenodo.org/records/8187288) | Noncommercial/research-only; code license does not license audio |
| `remfx-dynamicsuperb` | [Dynamic-SUPERB RemFX derivative](https://huggingface.co/datasets/DynamicSuperb/SoundEffectDetection_RemFx) | No separate clear audio license; research-only boundary |

`data/archive/` and `data/downloads/` contained duplicate/downloaded archives
of the records above (including IDMT, RemFX, EGFxSet, GuitarSet, Guitar-TECHS,
Aachen and ASRNN). Exact archive names, published byte sizes and hashes are in
`train/sources.json` and, where present, `remix/data_sources.json`. The local
`train/cache/` and `train/runs/` in the client checkout, and `runs/`,
`remix/runs/` and demo WAV/ZIP files in this checkout, were generated artifacts,
not independent publisher downloads. Their source/data lineage is in the cycle,
package, run and demo metadata where recorded; most excluded checkpoints and
exact generated audio cannot simply be re-downloaded. Reproduction would
require reacquiring eligible inputs, rerunning code, and revalidating gates.

This index intentionally does not claim that every locally generated bit is
recoverable. The owner approved permanent loss of local-only datasets,
checkpoints and demo audio after their provenance was documented.

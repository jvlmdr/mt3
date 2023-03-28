import logging
from pathlib import Path

from absl import app
from absl import flags
import librosa
import note_seq

import model

flags.DEFINE_string('model', 'mt3', 'Model {ismir2021, mt3}.')
flags.DEFINE_string('ckpt_dir', None, 'Path to model checkpoint dir. No symbolic links!', required=True)
flags.DEFINE_integer('sample_rate', 16000, 'Audio sample rate.')
flags.DEFINE_string('output_dir', './output/', 'Output dir.')

FLAGS = flags.FLAGS

# SAMPLE_RATE = 16000
# SF2_PATH = '../data/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'


def main(argv):
    input_files = argv[1:]
    if not input_files:
        logging.warning('no input files provided')
        return

    inference_model = model.InferenceModel(FLAGS.ckpt_dir, FLAGS.model)

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(0o755, parents=True, exist_ok=True)

    for input_file in input_files:
        input_file = Path(input_file)
        try:
            logging.info('load input: %s', input_file)
            audio = note_seq.audio_io.load_audio(input_file, sample_rate=FLAGS.sample_rate)
            logging.info('model inference')
            est_ns = inference_model(audio)
            output_file = output_dir / (input_file.stem + '.mid')
            logging.info('save to midi: %s', output_file)
            note_seq.sequence_proto_to_midi_file(est_ns, output_file)
        except Exception as ex:
            logging.warning('error encountered with file "%s": %s', input_file, ex)


if __name__ == '__main__':
    app.run(main)

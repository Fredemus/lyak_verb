/// the plan atm is to have this running in parallel with RayTrace
/// The end of the early reflections will be approximated somehow
/// and the output of this will be delayed with that many samples
// TODO: Pretty sure the delay is not perfectly robust (delay len won't be right when len gets lower and vec doesn't shrink)
// TODO: The schroeder allpass doesn't work at all for some reason? Maybe it does need the +x after all
use super::{Parameters, ABSORPTIONS, MAX_IR_LEN};
use std::f32::consts::{E, PI};
use std::sync::Arc;
pub struct AlgorithmicVerb {
    pub(crate) initial_delay: Delay,
    comb_filter_1: CombFilter,
    comb_filter_2: CombFilter,
    comb_filter_3: CombFilter,
    comb_filter_4: CombFilter,
    bass_combs: [CombFilter; 2],
    sc_ap1: SchroederAllpass,
    sc_ap2: SchroederAllpass,
    sc_ap3: SchroederAllpass,
    pub(crate) params: Arc<Parameters>,

    /// helps us get the reverb to decay at an appropriate speed for the room
    pub rt60: f32,
}
impl AlgorithmicVerb {
    pub fn process(&mut self, input: f32) -> f32 {
        // delaying this part of the reverb so it doesn't overlap (too much?) with RayTrace
        let x = self.initial_delay.process(input);

        let comb_out = self.comb_filter_1.process(x)
            + self.comb_filter_2.process(x)
            + self.comb_filter_3.process(x)
            + self.comb_filter_4.process(x)
            + (self.bass_combs[0].process(x)
            + self.bass_combs[1].process(x)) * 0.5;
        self.sc_ap3
            .process(self.sc_ap2.process(self.sc_ap1.process(comb_out)))
    }
    pub(crate) fn adjust(&mut self) {
        let x = self.params.length.get();
        let y = self.params.width.get();
        let z = self.params.height.get();
        self.set_rt60(x, y, z);
        self.set_comb_filters(x, y, z);
        self.adjust_feedback_filters();
    }
    /// sets the reverberation time according to sabine's formula
    fn set_rt60(&mut self, x: f32, y: f32, z: f32) {
        // area of left and right surfaces, since these are separated by x
        let lr_area = y * z;
        // and so on
        let fb_area = x * z;
        let tb_area = x * y;

        let absorption_area = lr_area * (ABSORPTIONS[0] + ABSORPTIONS[1])
            + fb_area * (ABSORPTIONS[2] + ABSORPTIONS[3])
            + tb_area * (ABSORPTIONS[4] + ABSORPTIONS[5]);

        let volume = x * y * z;

        self.rt60 = 0.161 * volume / absorption_area;
    }
    /// sets up the comb and allpass filters for the desired RT60
    fn set_comb_filters(&mut self, mut x: f32, mut y: f32, z: f32) {
        let fs = self.params.sample_rate.get();
        // 343 is speed of sound
        let sample_speed = fs / 343.;
        // it has been decided that the delay_lens for the comb_filters come from the distance between walls
        // however, to avoid delay lines being identical, some randomization happens (if they're equal?)
        // TODO: A bit of randomness for the coeffs here could be nice
        if (x - y).abs() < 0.02 {
            x = x * 0.95;
        }
        // since x shrinks twice if equal to both, this should be safe
        if (x - z).abs() < 0.02 {
            x = x * 0.95;
        }
        if (y - z).abs() < 0.02 {
            y = y * 1.05;
        }
        self.comb_filter_1.set_dly_len((x * sample_speed) as usize);
        self.comb_filter_2.set_dly_len((y * sample_speed) as usize);
        self.comb_filter_3.set_dly_len((z * sample_speed) as usize);
        // bass combs are meant to be very long delay lines, though still scaling somewhat with the room
        self.bass_combs[0].set_dly_len((x * y * z * sample_speed / 40.5 + 1413.) as usize);
        self.bass_combs[1].set_dly_len((x * y * z * sample_speed / 40. + 1201.) as usize);
        // the last comb filter doesn't have a good thing to be based on. For now just a weird ratio of comb_1,
        // for making the impulse denser. Based partly on diffusion. Gets denser the higher diffusion gets
        self.comb_filter_4.set_dly_len(
            (2. * x * sample_speed * (1. - self.params.diffusion.get() * 0.55)) as usize,
        );
        // setting the min delay len according to the  2048 - shortest comb delay (assumed to be 4)
        if self.comb_filter_4.dly_len < 2048 {
            self.initial_delay
                .set_dly_len(2048 - self.comb_filter_4.dly_len);
        } else {
            self.initial_delay.set_dly_len(1);
        }

        // feedbacks set to get the right RT60 according to
        // RT60 = (60/ (-20 * log(abs(feedback)))) * delay_len_ms
        let coeff = E.powf(-(3. * 10f32.ln()) / self.rt60);
        self.comb_filter_1.feedback = coeff.powf(self.comb_filter_1.dly_len as f32 / fs);
        self.comb_filter_2.feedback = coeff.powf(self.comb_filter_2.dly_len as f32 / fs);
        self.comb_filter_3.feedback = coeff.powf(self.comb_filter_3.dly_len as f32 / fs);
        self.comb_filter_4.feedback = coeff.powf(self.comb_filter_4.dly_len as f32 / fs);
        self.bass_combs[0].feedback = coeff.powf(self.bass_combs[0].dly_len as f32 / fs);
        self.bass_combs[1].feedback = coeff.powf(self.bass_combs[1].dly_len as f32 / fs);
    }
    pub fn adjust_feedback_filters(&mut self) {
        // this formula gives a nice curve between cutoff of 20 kHz and 5 kHz
        let fc = 5000. + 10000. * (1. - self.params.damping.get().powi(2));
        let fs = self.params.sample_rate.get();
        self.comb_filter_1.fb_filter.set_g(fc, fs);
        // terminate early if damping hasn't changed. Best way I could think of right now, but surely there must be
        // a better way to avoid calculating this needlessly
        if self.comb_filter_2.fb_filter.g == self.comb_filter_2.fb_filter.g {
            return;
        }

        self.comb_filter_2.fb_filter.set_g(fc, fs);
        self.comb_filter_3.fb_filter.set_g(fc, fs);
        self.comb_filter_4.fb_filter.set_g(fc, fs);
        self.comb_filter_4.fb_filter.set_g(fc, fs);
        self.comb_filter_4.fb_filter.set_g(fc, fs);
        self.bass_combs[0].fb_filter.set_g(fc, fs);
        self.bass_combs[1].fb_filter.set_g(fc, fs);
    }
    /// used for testing
    pub fn _get_dly_lens(&self) -> [usize; 6] {
        [
            self.comb_filter_1.dly_len,
            self.comb_filter_2.dly_len,
            self.comb_filter_3.dly_len,
            self.comb_filter_4.dly_len,
            self.bass_combs[0].dly_len,
            self.bass_combs[1].dly_len,
        ]
    }
}
impl Default for AlgorithmicVerb {
    fn default() -> AlgorithmicVerb {
        let mut a = AlgorithmicVerb {
            initial_delay: Delay {
                dly_len: MAX_IR_LEN,
                buffer: vec![0.; MAX_IR_LEN],
                ..Default::default()
            },
            comb_filter_1: CombFilter::default(),
            comb_filter_2: CombFilter::default(),
            comb_filter_3: CombFilter::default(),
            comb_filter_4: CombFilter::default(),
            bass_combs: [CombFilter::default(), CombFilter::default()],

            sc_ap1: SchroederAllpass::new(0.707, 1051),
            sc_ap2: SchroederAllpass::new(0.707, 337),
            sc_ap3: SchroederAllpass::new(0.707, 113),
            params: Arc::new(Parameters::default()),
            rt60: 1.,
        };
        a.adjust();
        a
    }
}
/// just a simple 2nd order lowpass filter with trapezoidal integration
pub struct FeedbackFilter2 {
    s0: f32,
    s1: f32,
    g: f32,
    // k = 1/Q-factor
    k: f32,
}
impl FeedbackFilter2 {
    pub fn new(fc: f32, fs: f32) -> FeedbackFilter2 {
        let mut a = FeedbackFilter2 {
            s0: 0.,
            s1: 0.,
            g: 0.,
            k: 1. / 0.5,
        };
        a.set_g(fc, fs);
        a
    }
    pub fn process(&mut self, input: f32) -> f32 {
        let g1 = 1. / (1. + self.g * (self.g + self.k));
        let g2 = self.g * g1;
        let v1 = g1 * self.s0 + g2 * (input - self.s1);
        let out = self.s1 + self.g * v1;

        self.s0 = 2. * v1 - self.s0;
        self.s1 = 2. * out - self.s1;
        out
    }
    /// using the bilinear transform
    pub fn set_g(&mut self, fc: f32, fs: f32) {
        self.g = ((PI * fc) / fs).tan();
    }
}
pub struct CombFilter {
    dly_len: usize,

    buffer: Vec<f32>,
    feedback: f32,
    index: usize,
    fb_filter: FeedbackFilter2,
}
impl Default for CombFilter {
    fn default() -> CombFilter {
        let mut a = CombFilter {
            dly_len: 100,
            buffer: vec![0.; 1000],
            feedback: 0.,
            index: 0,
            fb_filter: FeedbackFilter2::new(5000., 44100.),
        };
        a.set_dly_len(100);
        a
    }
}
impl CombFilter {
    fn set_dly_len(&mut self, len: usize) {
        if self.buffer.len() < len {
            self.buffer = vec![0.; len];
        }
        self.dly_len = len;
        self.index = len
    }
    // TODO: The initial delay doesn't work. Index probably has to start at same value as dly_len
    fn process(&mut self, input: f32) -> f32 {
        if self.index >= self.buffer.len() {
            self.index = 0
        }
        let out: f32;
        // this if-statement implements circularbuffer-like behavior
        // makes sure we iterate to the right sample
        if self.index < self.dly_len {
            out = self.buffer[(self.buffer.len() + self.index - self.dly_len)];
        } else {
            out = self.buffer[(self.index - self.dly_len)];
        }
        self.buffer[self.index] = input;

        // subtract output sample at some level to buffer here for negative feedback
        self.buffer[self.index] -= self.feedback * self.fb_filter.process(out);

        self.index += 1;
        out
    }
}

// TODO: replace index and big if-statement with read_index and write_index?
#[derive(Clone)]
pub struct Delay {
    dly_len: usize,

    buffer: Vec<f32>,
    feedback: f32,
    index: usize,
}

impl Delay {
    fn set_dly_len(&mut self, len: usize) {
        if self.buffer.len() < len {
            self.buffer = vec![0.; len];
        }
        self.dly_len = len;
        self.index = len
    }
    fn process(&mut self, input: f32) -> f32 {
        if self.index >= self.buffer.len() {
            self.index = 0
        }
        let out: f32;
        // this if-statement implements circularbuffer-like behavior
        // makes sure we iterate to the right sample
        if self.index < self.dly_len {
            out = self.buffer[(self.buffer.len() + self.index - self.dly_len)];
        } else {
            out = self.buffer[(self.index - self.dly_len)];
        }
        self.buffer[self.index] = input;
        // subtract output sample at some level to buffer here for negative feedback
        self.buffer[self.index] -= self.feedback * out;

        self.index += 1;
        out
    }

    /// for testing
    fn _process2(&mut self, input: f32) -> f32 {
        if self.index >= self.buffer.len() {
            self.index = 0
        }
        println!("index1: {}", self.index);
        // println!("index2: {}", self.index as i32 - self.dly_len as i32);

        let out: f32;
        // this if-statement implements circularbuffer-like behavior
        // makes sure we iterate to the right sample
        if self.index < self.dly_len {
            // if (self.buffer.len() as i32 + self.index as i32 - self.dly_len as i32) < 0 {
            //     println!("buffer len: {} index: {}, dly_len: {}, ", self.buffer.len(), self.index, self.dly_len)
            // }
            out = self.buffer[(self.buffer.len() + self.index - self.dly_len)];
            println!(
                "index2: {}",
                self.buffer.len() as i32 + self.index as i32 - self.dly_len as i32
            );
        } else {
            out = self.buffer[(self.index - self.dly_len)];
            println!("index2: {}", self.index - self.dly_len);
        }
        self.buffer[self.index] = input;

        // subtract output sample at some level to buffer here for negative feedback
        self.buffer[self.index] -= self.feedback * out;

        self.index += 1;
        out
    }
}
impl Default for Delay {
    fn default() -> Delay {
        let mut a = Delay {
            dly_len: 100,
            buffer: vec![0.; 1000],
            feedback: 0.,
            index: 0,
        };
        a.set_dly_len(100);
        a
    }
}
struct SchroederAllpass {
    delay: Delay,
    g: f32,
}
impl SchroederAllpass {
    fn new(g: f32, dly_len: usize) -> SchroederAllpass {
        let mut delay = Delay {
            dly_len,
            feedback: -g,
            ..Default::default()
        };
        delay.set_dly_len(dly_len);
        SchroederAllpass { delay, g }
    }
    fn process(&mut self, input: f32) -> f32 {
        let delay_out = self.delay.process(input);

        let out = delay_out * (1. - self.g.powi(2)) - input * self.g;

        out
    }
}

// use log_panics;
/// used to verify if delays work as intended
#[test]
fn test_delay() {
    // log_panics::init();
    let mut dly = Delay {
        dly_len: 2,
        buffer: vec![0.; 100],
        feedback: 0.5,
        ..Default::default()
    };
    let mut input_sample = 1.;
    let mut output_sample: f32;
    for i in 0..10 {
        output_sample = dly.process(input_sample);
        input_sample = 0.;
        println!("output at {}: {}", i, output_sample);
        // println!("index1: {}", self.buffer.len() + self.index - self.dly_len);
    }
}
#[test]
fn rt_60_test() {
    let x = 5.;
    let y = 4.;
    let z = 3.;
    // let _diff = 0.2;
    let mut algo_verb = AlgorithmicVerb::default();
    
    algo_verb.set_rt60(x, y, z);
    println!("rt60: {}", algo_verb.rt60);
}


#[test]
fn save_comb_filter_impulse() {
    // setting up the comb filter
    let x = 1.;
    let mut algo_verb = AlgorithmicVerb::default();
    algo_verb.set_comb_filters(x, x, x);
    algo_verb.comb_filter_1.fb_filter.set_g(20000., 44100.);
    algo_verb.comb_filter_2.fb_filter.set_g(20000., 44100.);
    // setting up hound for creating .wav files
    use hound;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(format!("testing/comb_impulse_{}m.wav", x), spec).unwrap();
    let len = 10000;
    let mut input_sample = 1.;
    // saving samples to wav file
    for _i in 0..len {
        writer
            .write_sample(algo_verb.comb_filter_1.process(input_sample))
            .unwrap();

        input_sample = 0.;
    }
}
#[test]
// #[ignore]
fn save_multicomb_filter_impulse() {
    // setting up the comb filter
    let x = 5.;
    let y = 4.;
    let z = 2.;
    let mut algo_verb = AlgorithmicVerb::default();
    algo_verb.set_comb_filters(x, y, z);
    // algo_verb.adjust_feedback_filters(20000., 44100.);
    algo_verb.comb_filter_1.fb_filter.set_g(20000., 44100.);
    algo_verb.comb_filter_2.fb_filter.set_g(20000., 44100.);
    algo_verb.comb_filter_3.fb_filter.set_g(20000., 44100.);
    algo_verb.comb_filter_4.fb_filter.set_g(20000., 44100.);
    // setting up hound for creating .wav files
    use hound;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(
        format!("testing/4combs_impulse_{}x{}x{}.wav", x, y, z),
        spec,
    )
    .unwrap();
    println!(
        "comb filter feedback: {} delay len: {}",
        algo_verb.comb_filter_1.feedback, algo_verb.comb_filter_1.dly_len
    );

    let len = 10000;
    // let mut impulse = vec![0.;len];
    let mut input_sample = 1.;
    // saving samples to wav file
    // for i in 0..impulse.len() {
    for _i in 0..len {
        // *i = algo_verb.comb_filter_1.process(input_sample);
        writer
            .write_sample(
                algo_verb.comb_filter_1.process(input_sample)
                    + algo_verb.comb_filter_2.process(input_sample)
                    + algo_verb.comb_filter_3.process(input_sample)
                    + algo_verb.comb_filter_4.process(input_sample),
            )
            .unwrap();

        input_sample = 0.;
    }
}
#[test]
fn save_schroeder_ap_impulse() {
    // setting up the comb filter
    let x = 1.;
    let mut algo_verb = AlgorithmicVerb::default();
    algo_verb.set_comb_filters(x, x, x);
    // setting up hound for creating .wav files
    use hound;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(format!("testing/schroeder_ap_all.wav"), spec).unwrap();

    let len = 10000;
    // let mut impulse = vec![0.;len];
    let mut input_sample = 1.;
    // saving samples to wav file
    for _i in 0..len {
        writer
            .write_sample(
                algo_verb.sc_ap3.process(
                    algo_verb
                        .sc_ap2
                        .process(algo_verb.sc_ap1.process(input_sample)),
                ),
            )
            .unwrap();
        input_sample = 0.;
    }
}
#[test]
fn compensating_comb_impulse() {
    // setting up the comb filter
    let x = 1.;
    let mut algo_verb = AlgorithmicVerb::default();
    algo_verb.set_comb_filters(x, 0.5 * x, x);
    algo_verb.comb_filter_1.fb_filter.set_g(20000., 44100.);
    algo_verb.comb_filter_2.fb_filter.set_g(20000., 44100.);
    algo_verb.comb_filter_2.feedback = algo_verb.comb_filter_1.feedback * 0.5;
    algo_verb.comb_filter_2.dly_len = algo_verb.comb_filter_1.dly_len / 2;

    // setting up hound for creating .wav files
    use hound;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(
        format!("testing/comb_ir_{}m_compensated_lower_fb.wav", x),
        spec,
    )
    .unwrap();

    let len = 10000;
    // let mut impulse = vec![0.;len];
    let mut input_sample = 1.;
    // saving samples to wav file
    // for i in 0..impulse.len() {
    for _i in 0..len {
        // *i = algo_verb.comb_filter_1.process(input_sample);
        writer
            .write_sample(
                algo_verb.comb_filter_1.process(input_sample)
                    + 2. * algo_verb.comb_filter_2.process(input_sample),
            )
            .unwrap();

        input_sample = 0.;
    }
    println!(
        "comb filter1 feedback: {} delay len: {}",
        algo_verb.comb_filter_1.feedback, algo_verb.comb_filter_1.dly_len
    );
    println!(
        "comb filter2 feedback: {} delay len: {}",
        algo_verb.comb_filter_2.feedback, algo_verb.comb_filter_2.dly_len
    );
}

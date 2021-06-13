/// Consider expanding the project to 4 dimensions for the hell of it lol

// TODO: refactor feedback filters so they use fc directly. The current way is a mess and constantly sets it unnecessarily
// TODO: Program still stalls on build_impulse() at times, albeit rarely. 
// Could fix/find problem by counting iterations of while loop 
#[macro_use]
extern crate vst;
use std::collections::VecDeque;
use std::sync::Arc;
use vst::buffer::AudioBuffer;
use vst::plugin::{Category, HostCallback, Info, Plugin, PluginParameters};
mod util;
use crate::util::{AtomicBool, AtomicF32};
mod algorithmic_verb;
mod raytrace;
// TODO: For now, early_ir is limited to 2048 samples. Is that reasonable?
/// 2048 samples chosen since it seems a reasonable upper bound for early reflections and would work well with
/// frequency domain convolution
const MAX_IR_LEN: usize = 2048;
/// the surfaces in absorptions follow same order as in raytrace:
///  left, right, front, back, top, bottom
const ABSORPTIONS: [f32; 6] = [0.02, 0.2, 0.02, 0.02, 0.1, 0.1];
struct LyakVerb {
    // Store a handle to the plugin's parameter object.
    params: Arc<Parameters>,
    raytrace: raytrace::RayTrace,
    algoverb: algorithmic_verb::AlgorithmicVerb,
    /// VecDeque instead of circular buffer here, since we need random access for convolution
    sample_buffer: VecDeque<f32>,
    damping_filter: algorithmic_verb::FeedbackFilter2,
    /// helps keep the reverb at a manageable level at all settings
    gain_factor: f32,
}

struct Parameters {
    length: AtomicF32,
    width: AtomicF32,
    height: AtomicF32,
    diffusion: AtomicF32,
    damping: AtomicF32,
    mix: AtomicF32,
    // TODO: How do we handle p_mic and p_lspk? not parameters for now
    sample_rate: AtomicF32,
    /// used for notifying process that the IR should be recalculated
    params_changed: AtomicBool,
    filter_changed: AtomicBool,
    highcut: AtomicF32,
}

impl Default for Parameters {
    fn default() -> Parameters {
        Parameters {
            length: AtomicF32::new(2.),
            width: AtomicF32::new(2.),
            height: AtomicF32::new(2.),
            diffusion: AtomicF32::new(0.2),
            damping: AtomicF32::new(0.),
            mix: AtomicF32::new(1.),
            sample_rate: AtomicF32::new(44100.),
            highcut: AtomicF32::new(15000.),
            params_changed: AtomicBool::new(false),
            filter_changed: AtomicBool::new(false),
        }
    }
}

impl Parameters {}

// member methods for the struct
impl LyakVerb {
    // naive time-domain convolution
    pub(crate) fn single_convolve(&self) -> f32 {
        let mut convolved = 0.;
        let k = self.raytrace.early_ir.len();
        for i in 0..k
        //  position in coefficients array
        {
            convolved += self.raytrace.early_ir[i] * self.sample_buffer[k - i];
        }
        return convolved;
    }
    // TODO: The system still ends up getting loud at extreme settings for room size,
    // presumably either because energy leaves the early_ir because of the size or more energy in late reflections,
    // should be fixed at some point
    /// used to avoid wet signals getting too loud when there's a lot of energy in the impulse response
    fn calc_gain_factor(&mut self) {
        let mut power = 0.;
        for i in 0..self.raytrace.early_ir.len() {
            power += self.raytrace.early_ir[i].powi(2);
        }
        // ref_power works pretty well to normalize the wet signal, but could potentially be fine tuned more
        let ref_power = 3.;

        let normalize_factor = ref_power / power;
        // println!("normalizing with factor: {}", normalize_factor);
        self.gain_factor = normalize_factor;
        // self.gain_factor = 0.0631;
    }
}

impl PluginParameters for Parameters {
    // get_parameter has to return the value used in set_parameter for presets to work correctly
    fn get_parameter(&self, index: i32) -> f32 {
        match index {
            0 => self.mix.get(),
            1 => self.highcut.get() / 10000. - 0.5,
            2 => (self.length.get() - 1.) / 9.,
            3 => (self.width.get() - 1.) / 9.,
            4 => (self.height.get() - 1.) / 9.,
            5 => self.diffusion.get() * 2.,
            6 => self.damping.get(),
            _ => 0.0,
        }
    }

    fn set_parameter(&self, index: i32, value: f32) {
        match index {
            0 => self.mix.set(value),
            1 => {
                self.highcut.set(5000. + value * 10000.);
            }
            2 => self.length.set(value * 9. + 1.),
            3 => self.width.set(value * 9. + 1.),
            4 => self.height.set(value * 9. + 1.),
            5 => self.diffusion.set(value * 0.5),
            6 => self.damping.set(value),
            _ => (),
        }
        // this is only necessary for some parameters, hence the if statement
        // TODO: the index < 5 can be used to avoid rebuilding impulse for damping, but might require another flag
        if index > 1
        /*&& index < 5*/
        {
            self.params_changed.set(true);
        }
        if index == 1 {
            self.filter_changed.set(true);
        }
    }

    fn get_parameter_name(&self, index: i32) -> String {
        match index {
            0 => "mix".to_string(),
            1 => "highcut".to_string(),
            2 => "length".to_string(),
            3 => "width".to_string(),
            4 => "height".to_string(),
            5 => "diffusion".to_string(),
            6 => "damping".to_string(),
            _ => "".to_string(),
        }
    }

    fn get_parameter_label(&self, index: i32) -> String {
        match index {
            0 => "%".to_string(),
            1 => "Hz".to_string(),
            2 => "m".to_string(),
            3 => "m".to_string(),
            4 => "m".to_string(),
            _ => "".to_string(),
        }
    }
    // This is what will display underneath our control.  We can
    // format it into a string that makes the most sense.
    fn get_parameter_text(&self, index: i32) -> String {
        match index {
            0 => format!("{:.1}", self.mix.get() * 100.),
            1 => format!("{:.0}", self.highcut.get()),
            2 => format!("{:.1}", self.length.get()),
            3 => format!("{:.1}", self.width.get()),
            4 => format!("{:.1}", self.height.get()),
            5 => format!("{:.3}", self.diffusion.get()),
            6 => format!("{:.3}", self.damping.get()),
            _ => format!(""),
        }
    }
}
impl Default for LyakVerb {
    fn default() -> Self {
        let params = Arc::new(Parameters::default());
        let mut raytrace = raytrace::RayTrace::default();
        let mut algoverb = algorithmic_verb::AlgorithmicVerb::default();
        algoverb.params = params.clone();
        raytrace.params = params.clone();
        let mut a = LyakVerb {
            params: params.clone(),
            raytrace: raytrace,
            algoverb: algoverb,
            sample_buffer: VecDeque::from(vec![0.; MAX_IR_LEN]),
            gain_factor: 1.,
            damping_filter: algorithmic_verb::FeedbackFilter2::new(
                15000.,
                params.sample_rate.get(),
            ),
        };
        a.calc_gain_factor();
        a
    }
}
impl Plugin for LyakVerb {
    fn new(_host: HostCallback) -> Self {
        let params = Arc::new(Parameters::default());
        let mut raytrace = raytrace::RayTrace::default();
        let mut algoverb = algorithmic_verb::AlgorithmicVerb::default();
        algoverb.params = params.clone();
        raytrace.params = params.clone();
        let mut a = LyakVerb {
            params: params.clone(),
            raytrace: raytrace,
            algoverb: algoverb,
            sample_buffer: VecDeque::from(vec![0.; MAX_IR_LEN]),
            gain_factor: 1.,
            damping_filter: algorithmic_verb::FeedbackFilter2::new(
                15000.,
                params.sample_rate.get(),
            ),
        };
        a.calc_gain_factor();
        a
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.params.sample_rate.set(rate);
        self.damping_filter
            .set_g(self.params.highcut.get(), self.params.sample_rate.get());
    }

    fn get_info(&self) -> Info {
        Info {
            name: "lyak_verb".to_string(),
            unique_id: 14289914,
            inputs: 1,
            outputs: 1,
            category: Category::Effect,
            parameters: 7,
            ..Default::default()
        }
    }

    fn process(&mut self, buffer: &mut AudioBuffer<f32>) {
        let mut wet: f32;
        for (input_buffer, output_buffer) in buffer.zip() {
            // we only need to rebuild the impulse response once per buffer and only if relevant parameters changed
            if self.params.params_changed.get() {
                self.params.params_changed.set(false);
                self.raytrace.set_planes();
                self.raytrace.build_impulse();
                self.calc_gain_factor();
                self.algoverb.adjust();
            }
            if self.params.filter_changed.get() {
                self.params.filter_changed.set(false);
                self.damping_filter
                    .set_g(self.params.highcut.get(), self.params.sample_rate.get());
            }

            let mix = self.params.mix.get();
            for (input_sample, output_sample) in input_buffer.iter().zip(output_buffer) {
                self.sample_buffer.push_back(*input_sample);
                // summing output of the 2 reverbs
                wet = self.single_convolve() + self.algoverb.process(*input_sample); // parallel
                // wet = self.single_convolve(); // only early reflections
                // wet = self.algoverb.process(self.single_convolve()); // series
                // wet = self.algoverb.process(*input_sample); // only late reflections

                // filtering the wet signal to tame the high frequencies a bit
                wet = self.damping_filter.process(wet);
                self.sample_buffer.pop_front();
                // mixing wet and dry signal
                *output_sample = wet * mix * self.gain_factor + *input_sample * (1. - mix);
            }
        }
    }

    fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        Arc::clone(&self.params) as Arc<dyn PluginParameters>
    }
}

plugin_main!(LyakVerb);
// simple test for debugging. used for checking if a variable has the expected value
// has to be run with cargo test -- --nocapture or the println! will be suppressed
#[test]
fn test_variable_value() {
    // for i in 0..20 {
    //     for j in 0..20 {
    //         println!("{}", 20 * i + j)
    //     }
    // }

    let mut plugin = LyakVerb::default();
    // for i in 0..6 {
    //     println!("plane: {:?}", plugin.raytrace.get_plane(i));
    // }
    plugin.params.length.set(5.);
    plugin.params.width.set(4.);
    plugin.params.height.set(3.);
    plugin.raytrace.set_planes();
    // for i in 0..6 {
    //     println!("plane: {:?}", plugin.raytrace.get_plane(i));
    // }
    plugin.raytrace.build_impulse();
    plugin.algoverb.adjust();
    // println!("before: {:?}", plugin.raytrace.early_ir);
    // println!("after: {:?}", plugin.raytrace.early_ir);

    println!("RT60: {}", plugin.algoverb.rt60);
    // let buffer = mut vst::buffer::AudioBuffer<'_, f32>
    // let mut input_sample = 1.;
    // let mix = plugin.params.mix.get();
    // println!("initial_delay: {} {} {}", plugin.algoverb.initial_delay.index);

    // let mut output_sample: f32;
    // for i in 0..5000 {
    //     // plugin.sample_buffer.push_back(input_sample);
    //     // // summing output of the 2 reverbs
    //     // let wet = plugin.single_convolve() + plugin.algoverb.process(input_sample);
    //     let wet = plugin.algoverb.process(input_sample);

    //     // plugin.sample_buffer.pop_front();
    //     // mixing wet and dry signal
    //     output_sample = wet * mix + input_sample * (1. - mix);
    //     println!("audio out: {}", output_sample);
    //     if wet.abs() > 0. {
    //         println!("sample at {}", i);
    //     }
    //     input_sample = 0.;
    // }
    // println!("impulse: {:?}", plugin.raytrace.early_ir);
}
#[test]
fn test_dly_lens() {
    let mut plugin = LyakVerb::default();
    // plugin.params.length.set(9.);
    // plugin.params.width.set(9.);
    // plugin.params.height.set(9.);
    plugin.params.length.set(8.5);
    plugin.params.width.set(6.9);
    plugin.params.height.set(6.1);
    println!("dly lens: {:?}", plugin.algoverb._get_dly_lens());
    plugin.algoverb.adjust();
    println!("dly lens: {:?}", plugin.algoverb._get_dly_lens());
}
// // used this test to fix an obscure error where it turned out that build_impulse never terminated
// should be run with cargo test --release -- --nocapture spot_the_crash_test > out
#[test]
fn spot_the_crash_test() {
    let mut plugin = LyakVerb::default();
    plugin.params.width.set(9.);
    plugin.params.height.set(9.);
    println!("from plugin: {}", plugin.params.height.get());
    println!("from raytrace: {}", plugin.raytrace.params.height.get());
    plugin.raytrace.set_planes();
    plugin.params.diffusion.set(0.50);
    plugin.params.damping.set(0.50);
    for i in 0..1000 {
        // rnd = rng.sample(Uniform::new(0., 1.));
        // plugin.params.length.set(1. + rnd * 9.);
        plugin.params.length.set(1. + i as f32 * 0.01 );
        plugin.raytrace.set_planes();
        println!("stalled at build_impulse");
        plugin.raytrace.build_impulse();
        // plugin.calc_gain_factor();
        // println!("stalled at adjust");
        // plugin.algoverb.adjust();
        println!("i: {}", i);
    }
    println!("dims: {:?}", [plugin.params.length.get(),plugin.params.width.get(),plugin.params.height.get()]);
}

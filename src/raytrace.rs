use super::{Parameters, ABSORPTIONS, MAX_IR_LEN};
use rand::distributions::Uniform;
use rand::Rng;
use std::f32::consts::{E, PI};
use std::sync::Arc;

fn dot_product(p1: &[f32], p2: &[f32]) -> f32 {
    p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
}
// Just a simple cross product with slices
fn _cross_product(p1: &[f32], p2: &[f32]) -> [f32; 3] {
    let c1 = p1[1] * p2[2] - p1[2] * p2[1];
    let c2 = p1[2] * p2[0] - p1[0] * p2[2];
    let c3 = p1[0] * p2[1] - p1[1] * p2[0];
    [c1, c2, c3]
}
// simple function for 3d distance between 2 points
#[inline(always)]
fn dist(x: &[f32; 3], y: &[f32; 3]) -> f32 {
    ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
}

// TODO: At some point the rounding errors get ridiculous. Might be a good idea to round the collision at the plane it hits

// These are only good in a very limited range, but that covers the range of shoot_rays
// Initial criterion benchmarking implies these are 4 times faster than the built-in functions
// found with maple: pade(sin(x), x, [6, 6])
#[inline(always)]
fn cheap_sin(x: f32) -> f32 {
    (1.000000000 * x - 0.1299565526 * x.powi(3) + 0.2903582076e-2 * x.powi(5))
        / (1.000000000
            + 0.3671011410e-1 * x.powi(2)
            + 0.6886010912e-3 * x.powi(4)
            + 0.7261929260e-5 * x.powi(6))
}
// found with maple: pade(sin(x), x, 10).
#[inline(always)]
fn cheap_cos(x: f32) -> f32 {
    1. - 0.5000000000 * x.powi(2) + 0.4166666668e-1 * x.powi(4) - 0.1388888889e-2 * x.powi(6)
        + 0.2480158730e-4 * x.powi(8)
        - 0.0000002755731922 * x.powi(10)
}
// TODO: IMPORTANT: It is definitely possible to add SIMD extensions, which would give a fat speed boost
// TODO: Raytrace needs to share params with AlgorithmicTail <- should be handled, but stays until verified
pub struct RayTrace {
    pub(crate) params: Arc<Parameters>,
    // rng: rand::prelude::ThreadRng,
    // Plane equations:
    // left, front and bottom will always be the same in this implementation
    left: [f32; 4],
    right: [f32; 4],
    front: [f32; 4],
    back: [f32; 4],
    top: [f32; 4],
    bottom: [f32; 4],

    // position of microphone and speaker
    p_lspk: [f32; 3],
    p_mic: [f32; 3],

    receiver_size: f32,
    w_limit: f32,
    air_absorb: f32,

    pub(crate) early_ir: [f32; MAX_IR_LEN],
}
impl Default for RayTrace {
    fn default() -> Self {
        let mut a = RayTrace {
            // Wasted allocation here, but whatever
            params: Arc::new(Parameters::default()),
            // left, front and bottom planes are always placed the same
            left: [1., 0., 0., 0.],
            front: [0., 1., 0., 0.],
            bottom: [0., 0., 1., 0.],
            // rest of the planes should be set with params, so just initialized as a 2x2x2 room
            right: [1., 0., 0., -(2. * 1.)],
            back: [0., 1., 0., -(2. * 1.)],
            top: [0., 0., 1., -(2. * 1.)],
            // these are also set by set_planes()
            p_lspk: [0., 0., 0.],
            p_mic: [0., 0., 0.],
            receiver_size: 0.2,
            w_limit: 0.01,
            // Found by absorption at 500 Hz: http://www.sengpielaudio.com/calculator-air.htm
            air_absorb: 1. - 0.9997,
            early_ir: [0.; MAX_IR_LEN],
        };
        a.set_planes();
        a.build_impulse();
        a
    }
}
impl RayTrace {
    // w_limit checks can most likely be omitted, since distance is more important limiting factor
    // ^ chosen not to be omitted and used to terminate with distance instead
    pub(crate) fn build_impulse(&mut self) {
        // speed of sound
        let c = 343.;
        self.early_ir = [0.; MAX_IR_LEN];
        let max_dist = (MAX_IR_LEN as f32 - 1.) * c / self.params.sample_rate.get();
        let mut w_ray: f32;
        let mut cum_dist: f32;
        let mut ray_dir: [f32; 3];

        // TODO: l,m,n doesn't need to be mutable if we remove the "for _x in 0..10" for loop
        let mut l: [f32; 400];
        let mut m: [f32; 400];
        let mut n: [f32; 400];
        // TODO: 10 is probably overkill here
        for _x in 0..3 {
            // Creating the rays
            let mut rng = rand::thread_rng();
            let rnd1: f32 = rng.sample(Uniform::new(0., 1.));
            let rnd2: f32 = rng.sample(Uniform::new(0., 1.));
            let rays = self.shoot_rays(rnd1, rnd2);
            l = rays.0;
            m = rays.1;
            n = rays.2;
            
            // shooting them
            for i in 0..399 {
                // used to fix a bug with a ray hitting the same plane multiple times,
                // causing a ray to fail (in a very cpu-intensive way)
                let mut last_plane = 10;
                ray_dir = [l[i], m[i], n[i]];
                let mut legal_planes = self.get_legal_planes(&ray_dir); // legal_planes type is [usize; 3]
                                                                        // the start point of the ray
                let mut start_point = self.p_lspk;
                // cumulative distance for the ray
                // backwards with distance of direct sound to avoid delays
                cum_dist = -dist(&self.p_mic, &self.p_lspk);
                // W_ray is the amplitude of the ray
                w_ray = 1.;
                while w_ray > self.w_limit {
                    // panic if the we end up in an infinite loop
                    // let legal_planes = [1,3,5];
                    // let legal_planes = [0,1,2,3,4,5];
                    for j in &legal_planes {
                        // for j in &legal_planes {
                        // break iteration (but not for loop) early if repeating last plane
                        if *j == last_plane {
                            continue;
                        }
                        let plane = self.get_plane(*j);
                        // find collision with a plane
                        let mut col_point =
                            self.find_collision_point(&start_point, &ray_dir, plane);
                        // check if the collision happens inside the room
                        if self.check_valid_collision(col_point) {
                            last_plane = *j;

                            // rounding the collision point at the coordinate of plane it hits,
                            // because the rounding errors get ridiculous
                            col_point = self.clamp_point(&col_point);
                            // calculating absorption. For now just by lowering W_ray according
                            // to absorption of the wall it hits
                            w_ray = w_ray * (1. - ABSORPTIONS[*j]);
                            // get a new ray_dir by reflection
                            // traveled by the ray

                            ray_dir = self.get_reflected_ray(
                                &ray_dir,
                                plane,
                                rng.sample(Uniform::new(0., 1.)),
                            );
                            // update legal planes for the new ray_dir
                            legal_planes = self.get_legal_planes(&ray_dir);
                            // add the traveled distance to the cumulative travelled
                            // distance
                            cum_dist = cum_dist + dist(&start_point, &col_point);
                            // breaking while loop early
                            // TODO: is there a better way? break will only break for loop it seems
                            if cum_dist > max_dist {
                                last_plane = 10;
                                w_ray = 0.;
                                break;
                            }
                            // update startpoint for the new ray
                            start_point = col_point;

                            // if the ray crosses the receiver
                            if self.check_receiver(ray_dir, col_point, self.p_mic) {
                                let delay = (cum_dist / c) * self.params.sample_rate.get();
                                self.early_ir[delay.round() as usize] =
                                    w_ray * E.powf(-cum_dist * self.air_absorb);
                                // going to next ray by resetting ray energy
                                last_plane = 10;
                                w_ray = 0.;
                            }
                            // breaking, since the current list of valid planes needs to be updated
                            break;
                        }
                    }
                }
            }
        }
        // self.normalize_ir();
    }

    fn get_legal_planes(&self, ray_dir: &[f32; 3]) -> [usize; 3] {
        let mut legal_planes = [0; 3];
        for i in 0..3 {
            if ray_dir[i] < 0. {
                legal_planes[i] = i * 2;
            } else {
                legal_planes[i] = i * 2 + 1;
            }
        }
        // legal_planes = [1, 2, 3, 4, 5, 6];
        legal_planes
    }
    fn _normalize_ir(&mut self) {
        let mut power = 0.;
        for i in 0..self.early_ir.len() {
            power += self.early_ir[i].powi(2);
        }
        // TODO: Tune ref_power to make it normalize better
        let ref_power = 3.;

        let normalize_factor = ref_power / power;
        // println!("normalizing with factor: {}", normalize_factor);
        for i in &mut self.early_ir {
            *i *= normalize_factor;
        }
    }
    fn shoot_rays(&self, rnd1: f32, rnd2: f32) -> ([f32; 400], [f32; 400], [f32; 400]) {
        let mut l = [0.; 400];
        let mut m = [0.; 400];
        let mut n = [0.; 400];
        let mut temp: f32;
        for i in 0..20 {
            for j in 0..20 {
                // moves around in a circle for 20 rays, then changing z and doing the same
                temp = 2.
                    * ((i as f32 + rnd1) / 20. - ((i as f32 + rnd1) / 20.).powi(2)).sqrt()
                    * (2. * PI * (j as f32 + rnd2) / 20.);
                l[20 * i + j] = cheap_cos(temp);
                m[20 * i + j] = cheap_sin(temp);
                n[20 * i + j] = 1. - 2. * (i as f32 + rnd1) / 20.;
            }
        }
        (l, m, n)
    }

    // How to pass i and j into this?
    // TODO: this version of shooting rays can potentially be used for optimization
    // fn gen_ray(rnd1: f32, rnd2: f32) -> [f32;3] {
    //     [2.*((i+rnd1)/20. - ((i+rnd1)/20.).powi(2)).sqrt() * (2.*PI*(j+rnd2)/20.).cos(), 2.*((i+rnd1)/20. - ((i+rnd1)/20.).powi(2)).sqrt() * (2.*PI*(j+rnd2)/20.).sin(), 1. - 2.*(i+rnd1)/20.]
    // }
    #[inline(always)]
    fn get_plane(&self, i: usize) -> &[f32; 4] {
        match i {
            0 => &self.left,
            1 => &self.right,
            2 => &self.front,
            3 => &self.back,
            4 => &self.bottom,
            5 => &self.top,
            _ => &[0.; 4],
        }
    }

    pub(crate) fn set_planes(&mut self) {
        let x = self.params.length.get();
        let y = self.params.width.get();
        let z = self.params.height.get();
        // it has been decided that z-coordinate of mic and speaker will be 1 meter. 
        // Rest is decided by size of room.
        // For now, mic and speakers get farther apart when room grows. Does that make sense?
        // might want to put some clipping on their spread
        self.p_lspk = [x / 2. + x / 5., y / 2. + y / 5., 1.];
        self.p_mic = [x / 2. - x / 5., y / 2. - y / 5., 1.];

        // the planes can be set quite simply since their normal vector stays the same
        // right plane
        self.right[3] = -(x * 1.);
        self.back[3] = -(y * 1.);
        self.top[3] = -(z * 1.);
    }


    // TODO: Should [f32;3] be a ref?
    fn get_reflected_ray(&self, ray_dir: &[f32; 3], plane: &[f32; 4], rnd1: f32) -> [f32; 3] {
        let mut new_ray_dir = [0.; 3];
        // ray direction is a normal specular reflection
        // formula (3.7) from paper
        let temp = 2.
            * ((ray_dir[0] * plane[0] + ray_dir[1] * plane[1] + ray_dir[2] * plane[2])
                // / (col_point.iter().map(|x| x.powi(2)).sum::<f32>()));
                / (plane[0].powi(2) + plane[1].powi(2) + plane[2].powi(2)));
        for i in 0..3 {
            new_ray_dir[i] = ray_dir[i] - temp * plane[i];
        }
        // here direction gets randomized, but the signs of the specular reflection is kept
        // TODO: This could probably be optimized by not instantiating a thread_rng here, but passing in 3 rnd vars instead?
        if rnd1 < self.params.diffusion.get() {
            let mut rng = rand::thread_rng();
            for i in 0..3 {
                new_ray_dir[i] = new_ray_dir[i].signum() * rng.sample(Uniform::new(0.0001, 1.));
            }
        }
        new_ray_dir
    }
    // This could potentially have the "how long does it travel through the mic" check, for more variance in impulse
    #[inline]
    fn check_receiver(&self, line: [f32; 3], point: [f32; 3], pos: [f32; 3]) -> bool {
        // some intermediate variables so the math looks less crazy
        let a0 = ((pos[0] - point[0]) * line[1] - (pos[1] - point[1]) * line[0]).powi(2);
        let a1 = ((pos[1] - point[1]) * line[2] - (pos[2] - point[2]) * line[1]).powi(2);
        let a2 = ((pos[2] - point[2]) * line[0] - (pos[0] - point[0]) * line[2]).powi(2);
        let d = ((a0 + a1 + a2) / (line.iter().map(|x| x.powi(2)).sum::<f32>())).sqrt();
        // receiver is a 0.2m sphere atm
        if d < self.receiver_size {
            return true;
        }
        false
    }
    // Is this the best way to find collision point? Could potentially use crossproduct function instead
    fn find_collision_point(
        &self,
        startpoint: &[f32; 3],
        ray_dir: &[f32; 3],
        plane: &[f32; 4],
    ) -> [f32; 3] {
        let d = plane[3];
        let mut col_point = [0.; 3];
        let time =
            (startpoint[0] * plane[0] + startpoint[1] * plane[1] + startpoint[2] * plane[2] + d)
                / (ray_dir[0] * plane[0] + ray_dir[1] * plane[1] + ray_dir[2] * plane[2]);
        for i in 0..3 {
            col_point[i] = startpoint[i] - time * ray_dir[i];
        }
        col_point
    }
    // clamps a point to be inside the room
    fn clamp_point(&self, point: &[f32; 3]) -> [f32; 3] {
        let dims = [self.params.length.get(), self.params.width.get(), self.params.height.get()];
        let mut clamped_point = [0.; 3]; 
        for i in 0..3 {
            if point[i] < 0. {
                clamped_point[i] = 0.;
            }
            else if point[i] > dims[i] {
                clamped_point[i] = dims[i];
            }
            else {
                clamped_point[i] = point[i];
            }
        }
        clamped_point
    }
    /// this is a naive check that will only work for the simple square room
    /// the naive check was chosen since it's faster
    #[inline(always)]
    fn check_valid_collision(&self, col_point: [f32; 3]) -> bool {
        let offset = 0.1;
        if col_point[0] >= -offset && col_point[0] <= self.params.length.get() + offset {
            if col_point[1] >= -offset && col_point[1] <= self.params.width.get() + offset {
                if col_point[2] >= -offset && col_point[2] <= self.params.height.get() + offset {
                    return true;
                }
            }
        }
        false
    }
}
#[test]
fn plane_check() {
    // let mut raytrace = RayTrace::default();
    // raytrace.params.length.set(5.);
    // raytrace.set_planes();
    // for i in 0..6 {
    //     println!("plane {}: {:?}", i, raytrace.get_plane(i));
    // }
    // println!("is here");
    // println!("legal planes : {:?}", raytrace.get_legal_planes(&ray_dir));
    // let plane = raytrace.get_plane(1);
    // assert!(plane[0] * raytrace.params.length.get() + plane[1] * raytrace.params.width.get() + plane[2] * raytrace.params.height.get() + plane[3] ==0.,"fuck");
    let legal_planes = [0, 2, 4];
    for i in 0..6 {
        if legal_planes.contains(&i) {
            println!("i: {}", i);
        }
    }
}
#[test]
fn collision_check() {
    let raytrace = RayTrace::default();
    raytrace.params.length.set(3.3899999);
    raytrace.params.width.set(9.);
    raytrace.params.height.set(9.);
    let start_point = [2.0044425, 9.0, 9.0];

    println!("{}", raytrace.check_valid_collision(start_point));
}
// TODO: Test find_collision function here?
// #[test]
// fn collision_check2() {
//     let raytrace = RayTrace::default();

// }
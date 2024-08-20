#version 450

in vec4 gl_FragCoord;
layout(location = 0) out vec4 frag_color;
layout(set = 0, binding = 0) uniform Uniforms {
	int window_size_physical_x;
	int window_size_physical_y;
	float mousex;
	float mousey;
};

// TODO: gamma correct
// TODO: antialiasing

// some inspired by http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
// mult by recip faster?

// consts

#define PI 3.1415926538
#define PIf2 1.570796327 // pi fraction 2 (pi/2)
#define PI2 6.283185307 // 2*pi

const float inf = 1.0/0.0;

const int k_ray_marching_steps = 30;
const float k_ray_marching_start_offset = 0.0001;
const float k_ray_marching_max_dist = 100.0;
const float k_ray_marching_epsilon = 0.0001;
const float k_derivative_epsilon = 0.0001;
const int k_reflection_bounces_max = 6;
const int k_iter_count = 1;
#define k_fib_diffuse 0
#define k_fib_diffuse_count 10 // should be higher than k_iter_count
#define k_plastic_use 1
#define k_plastic 0.08
#define k_rnd_diffuse_use 1
#define k_rnd_diffuse 0.1 // 0 - 0.02 - 5.0

const vec2 k_v2 = vec2(1.0,1.0);
const vec3 k_i = vec3(1.,0.,0.);
const vec3 k_j = vec3(0.,1.,0.);
const vec3 k_k = vec3(0.,0.,1.);
#define vec3zero vec3(0.,0.,0.)


// external

// https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}
// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }
// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}
// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }


// structs

struct Material {
	vec3 color;
	vec3 emission;
	float reflection;
	float diffusion;
};
struct SceneResult {
	float dist;
	Material material;
};


// utils

mat4 rotation_matrix_for_angles(in vec3 angles) {
	vec3 s = sin(angles);
	vec3 c = cos(angles);
	mat4 xrot = mat4(1.,0.,0.,0.,    0.,c.x,-s.x,0., 0.,s.x,c.x,0.,  0.,0.,0.,1.);
	mat4 yrot = mat4(c.y,0.,s.y,0.,  0.,1.,0.,0.,    -s.y,0.,c.y,0., 0.,0.,0.,1.);
	mat4 zrot = mat4(c.z,-s.z,0.,0., s.z,c.z,0.,0.,  0.,0.,1.,0.,    0.,0.,0.,1.);
	return xrot*yrot*zrot;
}

float rot_angle(in float a, in float a2) {
	return mod(a - a2, PI2);
}
// returns in 0..<2PI?
float angle_of_vec2(in vec2 v) {
	// https://en.wikipedia.org/wiki/Atan2
	// return atan(v.y/v.x)+PIf2;
	return atan(-v.y, -v.x)+PI;
	// return acos(dot(v, vec2(1.0, 0.0))/length(v));
	// float a = atan(v.y/v.x);
	// if (a < 0.0) a += PI;
	// if (v.y < 0.0) a += PI;
	// return a;
}

vec2 rot2(in vec2 v, in float a) {
	// [1, 0] rot PIf2 = [0, 1] = [xc, xs]
	// [0, 1] rot PIf2 = [-1, 0] = [-ys, yc]
	// [x, y] rot a = [(c, -s) (s, c)][x, y] = [xc-ys, xs+yc]
	float s = sin(a);
	float c = cos(a);
	return vec2(v.x*c-v.y*s, v.x*s+v.y*c);
}
vec3 rotate_by_angles(in vec3 v, in vec3 a) {
	return (rotation_matrix_for_angles(a)*vec4(v, 1.0)).xyz;
}

vec3 rotation_angles_between(in vec3 a, in vec3 b) {
	return vec3(
		angle_of_vec2(b.yz)-angle_of_vec2(a.yz),
		angle_of_vec2(b.xz)-angle_of_vec2(a.xz),
		angle_of_vec2(b.xy)-angle_of_vec2(a.xy)
	);
}



// h=0..PI2,s,b is 0..=1
vec3 hsb(in float h, in float s, in float b) {
	// TODO: not checked

	s = 1-pow(1-s, 2); // makes it smoother, but slower?

	// map h from +klyka..0..-klyka -> 0..1..0
	// = map (rot h -klyka) from 2klyka..klyka..0 -> 0..1..0
	// = map abs((rot h -klyka)-klyka) from klyka..0 -> 0..1
	// = smoothstep abs((rot h -klyka)-klyka) from klyka..0
	float klyka = PI2/3.;
	return mix(vec3(1.,1.,1.), normalize(vec3(
		smoothstep(klyka, 0., abs((rot_angle(h, -klyka)-klyka))),
		smoothstep(klyka, 0., abs((rot_angle(h, -klyka+klyka)-klyka))),
		smoothstep(klyka, 0., abs((rot_angle(h, -klyka+klyka*2.)-klyka)))
	)), s)*b;

}
// h=0..PI2
vec3 hue_to_rgb(in float h) {
	return hsb(h, 1.0, 1.0);
}

const float golden_ratio = (1.0 + sqrt(5.0))*0.5;
const float golden_ratio_recip = 1.0/golden_ratio;
// vec2 p = vec2(i*(1.0/n), i*golden_ratio_recip);
// // p.y = p.y - floor(p.y*(1.0/grid_height))*grid_height; // for grid (origin: 0,0 size: 1,1)
// p = rot2(vec2(sqrt(p.x), 0.), PI2*p.y); // r: 0..1


// constructors

Material Material_hs(in float hue, in float saturation) {
	return Material(hsb(hue, saturation, 1.0), vec3zero, 1.0, 0.);
}
Material Material_blend(in Material base, in Material other, in float percent_other) {
	return Material(
		mix(base.color, other.color, percent_other),
		mix(base.emission, other.emission, percent_other),
		mix(base.reflection, other.reflection, percent_other),
		mix(base.diffusion, other.diffusion, percent_other)
	);
}
// const Material Material_white = Material(vec3(1.,1.,1.), vec3zero, 0., 0.);
// #define Material_white Material(vec3(1.,1.,1.), vec3zero, 0., 0.)
Material Material_white_get() {
	return Material(vec3(1.,1.,1.), vec3zero, 0., 0.);
}
Material Material_green_get() {
	return Material(vec3(0.,1.,0.), vec3zero, 0., 0.);
}


// all sdf's are from origin?

float plane_sdf(in vec3 p, in vec3 n, in float r) {
	return dot(n, p) - r;
}

float sphere_sdf(in vec3 p, in float r) {
	return length(p) - r;
}

// size is halfsize (from center to one corner)
float box_sdf(in vec3 p, in vec3 size) {

	// dxy = d.xy
	// d_xy = any dxy < 0.0 ? max dxy : len dxy; dmz = (d_xy, d.z)
	// d_mz = any dmz < 0.0 ? max dmz : len dmz; return d_mz

	//    | S : C
	//    | - s ....
	//    | I | S  d
	// -- c --------
	//    |        p
	// I: inside box
	// s: size (positive point relative to c)
	// c: center
	// S: Side field of box
	// C: Corner field of box
	// p: asking point
	// d: distance/vector from corner to "canonicalized" p
	// ---
	// center is origo: p -= c;
	// symmetric, convert to first quadrant: p = abs(p) // "canonicalized"
	// d = s->p;
	vec3 d = abs(p) - size;
	// if d is in I, all of d is negative, distance closest to edge is the maximum component of d
	// if d is in S, some of d is negative, at least one is positive, closest distance to box is the maximum component of d
	// if d is in C, all of d is positive, closest distance to box is the distance to the corner = length of d = length of d_pos
	// float d_max = max(max(d.x, d.y), d.z);
	// vec3 d_pos = max(d, 0.0);
	// return length(d_pos) + min(d_max, 0.0);
	// though not fully correct; in S, value will be too large? (both terms contribute?)
	return length(max(d, 0.0)) + min(max(max(d.x, d.y), d.z), 0.0);
}

void union_sdf(inout SceneResult base, in SceneResult added) {
	if (added.dist <= base.dist) {
		base = added;
	}
}

// try k=0.1
void union_smooth_sdf(inout SceneResult base, in SceneResult added, in float k) {
	float h = max(0, k-abs(base.dist-added.dist))/k;
	float dist = min(base.dist, added.dist) - h*h*k*(1.0/4.0);

	// dist = base.dist, use base.material
	// dist = added.dist, use added.material
	// dist is < base.dist, added.dist
	// if closer to base, it will be less then base, added, but will be closer to base
	// if same dist to both, blend should be 50%
	float bd = base.dist-dist;
	float ad = added.dist-dist;
	float percent_added = bd/(bd+ad); // 1-ad/(bd+ad)

	// percent_added = 0.;
	// dist = min(base.dist, added.dist);

	base = SceneResult(dist, Material_blend(base.material, added.material, percent_added));
}
void sub_smooth_sdf(inout SceneResult base, in SceneResult added, in float k) {
	float h = max(0, k-abs(base.dist-added.dist))/k;
	float dist = max(base.dist, -added.dist) + h*h*k*(1.0/4.0);

	// dist = base.dist, use base.material
	// dist = added.dist, use added.material
	// dist is < base.dist, added.dist
	// if closer to base, it will be less then base, added, but will be closer to base
	// if same dist to both, blend should be 50%
	float bd = base.dist-dist;
	float ad = added.dist-dist;
	float percent_added = bd/(bd+ad); // 1-ad/(bd+ad)

	base = SceneResult(dist, base.material
	// use for inside material?
	// Material(mix(base.material.color, added.material.color, percent_added))
	);
}
void union_smooth_sdf(inout SceneResult base, in SceneResult added) {
	union_smooth_sdf(base, added, 0.1);
}


SceneResult scene_sdf_res_1(in vec3 p) {
	const Material white = Material_white_get();
	SceneResult r = SceneResult(inf, white);

	Material blue_ball_mat = Material(hsb(PI2*0.5, 1., 1.0), vec3zero, 1.0, 1.0);
	Material green_ball_mat = Material(hsb(PI2*0.3, 1., 1.0), vec3zero, 1.0, 1.0);
	union_sdf(r, SceneResult(sphere_sdf(p-vec3(0.3, -0.3, -0.3), 0.23), blue_ball_mat));
	union_sdf(r, SceneResult(sphere_sdf(p-vec3(-0.3, -0.3, -0.3), 0.23), green_ball_mat));

	return r;
}


SceneResult scene_sdf_res(in vec3 p) {
	// return scene_sdf_res_1(p);
	SceneResult r1 = scene_sdf_res_1(p);

	const Material white = Material_white_get();
	SceneResult r = SceneResult(inf, white);
	// r = union_sdf(r, sphere_sdf(p, mousex));
	// r = union_sdf(r, sphere_sdf(p-vec3(.6,0.3,0.2), 0.3));
	// r = union_smooth_sdf(r, sphere_sdf(p-vec3(-.6,0.3,0.2), 0.3), 0.3);

	vec2 mp = vec2((mousex-0.5)*2.0*1.4, -2.0*(mousey-0.5)*1.1);
	// mp = vec2(0.5, 0.1);
	// mp = vec2(0.0,-0.3);

	// r = union_smooth_sdf(r, sphere_sdf(p- vec3(-0.3, -0.3, -0.5), 0.23), 0.1);

	// floor
	// float offset = sin(p.x*10.0)*0.05;
	// offset += sin(p.z*5.0)*0.05;
	// vec3 pos = vec3(0.0,-1.0 + offset, 0.0);
	// vec3 dir = vec3(0.0, 1.0, 0.0);

	// float thickness = 0.0;
	// float plane_dist = plane_sdf(
	// 	p - pos,
	// 	normalize(dir),
	// 	thickness
	// );

	// r = union_sdf(r, plane_dist);

	// walls
	union_smooth_sdf(r, SceneResult(plane_sdf(p-vec3(-1.0,0.0,0.0), normalize(vec3(1.0, 0.0, 0.0)), 0.), white));
	union_smooth_sdf(r, SceneResult(box_sdf(p-vec3(1.0+0.01,0.0,0.0), vec3(0.02,10.,10.)), white));
	union_smooth_sdf(r, SceneResult(plane_sdf(p-vec3(0.0,1.0,0.0), normalize(vec3(0.0, -1.0, 0.0)), 0.), white));
	union_smooth_sdf(r, SceneResult(plane_sdf(p-vec3(0.0,0.0,-2.0), normalize(vec3(0.0, 0.0, 1.0)), 0.), white));
	//union_smooth_sdf(r, SceneResult(plane_sdf(p-vec3(0.0,0.0,6.0), normalize(vec3(0.0, 0.0, -1.0)), 0.), white));

	// sphere cutout
	// sub_smooth_sdf(r, SceneResult(sphere_sdf(p-vec3(mp.xy, -0.4), 0.22), Material_hs(2., 1.)), 0.2);

	SceneResult rballs = SceneResult(inf, white);

	// TODO: make waves radial instead?
	float wave_offset = 0.0;
	wave_offset += sin(p.x*1.1*10.0)*0.01;
	wave_offset += sin(p.z*0.7*10.0)*0.02;
	union_sdf(rballs, SceneResult(plane_sdf(p-vec3(0.0,-1.0+wave_offset,0.0), normalize(vec3(0.0, 1.0, 0.0)), 0.), Material_hs(3.33, 0.5)));

	// camera inside sphere?
	// r = union_smooth_sdf(r, max(sphere_sdf(p- vec3(0.0,0.0,0.0), 3.0), -sphere_sdf(p- vec3(0.0,0.0,0.0), 2.5)), 0.3);


	// fibonacci balls
	// #define n_max 28
	// for (int i = 0; i<n_max; i++) {
	// 	float n = n_max*0.4;
	// 	vec2 p2 = vec2(i*(1.0/n), i*golden_ratio_recip);
	// 	float r = sqrt(p2.x);
	// 	float theta = PI2*p2.y;
	// 	p2 = rot2(vec2(r, 0.), theta);
	// 	float dm = random(p2)*0.05+0.1;
	// 	SceneResult bsr = SceneResult(sphere_sdf(p-vec3(p2*-0.43, -0.5+random(p2)*0.6), dm),
	// 		Material(hsb(i/n*PI2*8.7, 0.9, 1.0), vec3(smoothstep(13.,15., i)), 1.0));

	// 	sub_smooth_sdf(rballs, bsr, dm/1.2);
	// 	union_sdf(rballs, bsr);
	// }

	//union_smooth_sdf(rballs, SceneResult(sphere_sdf(p-vec3(-0.3, -0.3, -0.5), 0.23), Material_hs(0., 1.)));


	Material blue_ball_mat = Material(hsb(PI2*0.5, 1., 1.0), vec3zero, 1.0, 1.0);
	Material red_ball_mat = Material(hsb(PI2*0.1, 1., 1.0), vec3zero, 1.0, 1.0);
	SceneResult ball1 = SceneResult(sphere_sdf(p-vec3(0.3, -0.3, -0.3), 0.23), blue_ball_mat);
	SceneResult ball2 = SceneResult(sphere_sdf(p-vec3(-0.3, -0.3, -0.3), 0.23), red_ball_mat);
	//union_sdf(rballs, ball1);
	//union_sdf(rballs, ball2);
	union_sdf(rballs, ball2);
	union_smooth_sdf(rballs, ball1, 0.6);
	// union_smooth_sdf(rballs, SceneResult(sphere_sdf(p-vec3(0.3, -0.3, -0.3), 0.23), blue_ball_mat), 0.8);
	union_smooth_sdf(rballs, SceneResult(sphere_sdf(p-vec3(-mp.xy, -0.4)*1.5, 0.23*2.), Material_hs(2.+mp.x, 1.)));
	// union_smooth_sdf(rballs, SceneResult(box_sdf(p-vec3(mp.xy, -0.4), vec3(0.23*2., 0.23*2., 0.23*2.)), Material_hs(2.+mp.x, 1.)), 0.9);
	//union_sdf(r, rballs);
	// union_smooth_sdf(r, rballs, 0.2);

	// lamps
	{
		float repetition_z_interval = 0.6;
		float lamp_h = 0.08;
		vec3 lamp_s = vec3(.6,.02,.13);
		vec3 lamp_p = vec3(0.,1.0,0.22+floor(p.z/repetition_z_interval)*repetition_z_interval);
		// lamp_p.y -= (p.z-lamp_p.z)*0.8;
		// abs trick for symmetry
		vec3 foot_p = p-lamp_p-vec3(0.,-lamp_h+lamp_h/2.0,0.);
		// foot_p.x = abs(foot_p.x)-lamp_s.x*0.91;
		// foot_p.z = abs(foot_p.z)-lamp_s.z*0.91;
		foot_p.xz = abs(foot_p.xz)-lamp_s.xz*0.91;
		union_sdf(r, SceneResult(box_sdf(foot_p, vec3(.03, lamp_h/2., .03)), white));

		union_sdf(r, SceneResult(box_sdf(p-lamp_p-vec3(0.,-lamp_h+0.008,0.), (lamp_s+vec3(0.03, .0, 0.03)-vec3(0.02)))-0.02, Material(hsb(0., 0., 1.), vec3(0.), 0.4, 0.)));
		union_sdf(r, SceneResult(box_sdf(p-lamp_p-vec3(0.,-lamp_h,0.), lamp_s), Material(hsb(0., 0., 1.), vec3(1.), 0., 0.)));
	}

	//union_sdf(r, rballs);
	union_smooth_sdf(r, rballs, 0.4);
	//union_sdf(r, r1);
	

	// mirror
	// union_sdf(r, SceneResult(box_sdf(p-vec3(-0.7,0.,.12), vec3(0.015,.3,.3)-0.01)-0.01, Material(hsb(PI2/3., 0., 1.), vec3(0.), .8)));
	// red ball platform
	// union_smooth_sdf(r, SceneResult(box_sdf(p-vec3(-0.3, -0.3-0.26-0.015/2, -0.5), vec3(.26,0.015,.26)-0.01)-0.01, Material(hsb(PI2/3., 0., 1.), vec3(0.), .1)), 0.05);

	// window cutout
	// sub_smooth_sdf(r, SceneResult(box_sdf(p-vec3(1.,0.,.82), vec3(0.8,.3,.6)-0.01)-0.01, Material(hsb(PI2/3., 0., 1.), vec3(0.), .8)), 0.1);


	// union_sdf(r, SceneResult(sphere_sdf(p-vec3(0.,1.,0.), 0.23), Material(hsb(0., 0., 1.), vec3(1.,1.,1.))));

	// r = union_smooth_sdf(r, sphere_sdf(p- vec3(mp.xy, -0.4), 0.23), 0.3);
	// r = union_sdf(r, sphere_sdf(p-xy*0.5, 0.1));
	// r = union_sdf(r, sphere_sdf(p-vec3(0.5), 0.1));

	//r = union_sdf(r, box_sdf(p, vec3(-0.2, -0.2, -0.7), vec3(0.2, 0.1, 0.01)));
	// r = union_smooth_sdf(r, box_sdf(p, vec3(0.5, -0.7, -0.4), vec3(0.1, 0.05, 0.2)), 0.15);
	// r = union_smooth_sdf(r, box_sdf(p-vec3(mp.xy, -0.4), vec3(0.5, 0.05, 0.05)), 0.35);
	// r = union_smooth_sdf(r, box_sdf(p-vec3(mp.xy, mp.y*1.2), vec3(0.02, 0.02, 0.4)), 0.15);

	return r;
}
float scene_sdf(in vec3 p) {
	return scene_sdf_res(p).dist;
}

vec3 scene_color(in vec3 p) {
	Material material = scene_sdf_res(p).material;
	return material.color;
}



// ray marching

vec3 scene_norm(in vec3 p) {
	// normal is perpendicular to tangent
	// tangent's tilt is the derivative
	// derivative is dy/dx
	// dy/dx is lim(h->0) of f(a+h)-f(a) / h
	// in multi-variable, derivative is called gradient
	// normalizing the gradient, we (almost?) get the normal?

	float h = k_derivative_epsilon;
	float o = scene_sdf(p);
	vec3 grad = vec3(
		scene_sdf(p+vec3(h,.0,.0)),
		scene_sdf(p+vec3(.0,h,.0)),
		scene_sdf(p+vec3(.0,.0,h))
	) - o;
	// grad = vec3(
	// 	scene_sdf(p+vec3(h,.0,.0))-scene_sdf(p-vec3(h,.0,.0)),
	// 	scene_sdf(p+vec3(.0,h,.0))-scene_sdf(p-vec3(.0,h,.0)),
	// 	scene_sdf(p+vec3(.0,.0,h))-scene_sdf(p-vec3(.0,.0,h))
	// );

	return normalize(grad);
}

struct RayMarchRes {
	float dist;
	float accumulated_closeness;
	Material material;
};

// ray origin, ray direction
RayMarchRes do_raymarch(in vec3 ro, in vec3 rd) {
	
	// for antialiasing
	// float first_dist_within_1px = -1.0;
	// float first_dist_within_1px_p = -1.0;

	float closeness_f = 0.0005;
	float acc_closeness = 0;
	Material last_material = Material_white_get();

	float dist = 0;
	float step_dist;
	ro += k_ray_marching_start_offset*rd;
	for (int i=0; i<k_ray_marching_steps; i++) {

		// for antialiasing
		// TODO; also, should be from closest raster pixel edge?
		// float _1px_dist_at_z = 0.001;

		SceneResult res = scene_sdf_res(ro);
		last_material = res.material;
		step_dist = res.dist;
		dist += step_dist;
		acc_closeness += closeness_f/step_dist;

		// for antialiasing
		// if (step_dist <= _1px_dist_at_z && first_dist_within_1px_p<0.0) {
		// 	first_dist_within_1px = dist;
		// 	first_dist_within_1px_p = step_dist/_1px_dist_at_z;
		// }

		if (step_dist <= k_ray_marching_epsilon) break;
		if (dist >= k_ray_marching_max_dist) break;
		ro += step_dist*rd;
	}

	// for antialiasing
	// if (first_dist_within_1px_p<0.0) {
	// 	first_dist_within_1px = dist;
	// 	first_dist_within_1px_p = 0.0;
	// }

	// return vec3(dist, first_dist_within_1px, first_dist_within_1px_p);
	return RayMarchRes(dist, acc_closeness, last_material);
}


vec4 simplified_lighting(in vec3 p, in RayMarchRes rm, in vec3 eye_pos, in vec3 ray_dir) {
	// see phong lighting model
	// assume rm.dist < k_ray_marching_max_dist

	vec3 norm = scene_norm(p);

	float ambient_lighting = 0.008;

	float c = 0;
	c += ambient_lighting;

	vec3 from_eye = normalize(p-eye_pos);
	vec3 to_eye = -from_eye;

	float mousex2 = 0.2;//mix(0.72, 0.5, rm.material.diffusion);

	{
		// vec3 light_pos = vec3(0.3, 1.3,(mousex-0.5)*3.0);
		vec3 light_pos = vec3(0., 0., 0.);
		// vec3 light_pos = vec3((mousex-0.5)*2.0, -2.0*(mousey-0.5), 0.0);
		float intensity = 0.5;

		vec3 from_light = normalize(p-light_pos);
		vec3 perfect_reflection = reflect(from_light, norm);

		float cos_angle_to_perfect = dot(perfect_reflection, norm);
		float angle_to_perfect_10 = 1-min(1, (acos(cos_angle_to_perfect)/PIf2));
		c += angle_to_perfect_10*intensity;

		float specular_intensity = 0.005/pow(mousex2*10.0, 10.0);
		float specular_shininess = 5.5*mousex2*10.0;

		float cos_angle_eye_to_reflection = dot(perfect_reflection, from_eye);
		float angle_eye_to_reflection_10 = acos(cos_angle_eye_to_reflection)/PIf2;
		c += pow(angle_eye_to_reflection_10, specular_shininess)*specular_intensity;
	}

	vec4 color = vec4(c, c, c, 1.0);
	return color;
}

vec4 trace_color(in vec3 eye_pos, in vec3 ray_dir, in RayMarchRes rm, in int iter_nr, in int iter_count, in float seed) {

	vec4 color = vec4(0.);
	vec4 col_filter = vec4(1.);
	// vec4 reflection_filter = vec4(1.);

	float is_boll = rm.material.reflection;

	for (int i = 0; i<k_reflection_bounces_max; i++) {
		// if (rm.dist >= k_ray_marching_max_dist) {
		// 	// sky/background?
		// 	// float c = (uv.y+0.5+0.2)*0.04;
		// 	// frag_color = vec4(0.0,c*0.2,c, 1.0);
		// 	// vec4(0.);
		// 	col_filter = vec4(0.);
		// 	break;
		// }
		col_filter *= smoothstep(k_ray_marching_max_dist, k_ray_marching_max_dist-k_derivative_epsilon, rm.dist);
		
		vec3 p = eye_pos + ray_dir*rm.dist;
		vec3 norm = scene_norm(p);
		vec3 from_eye = normalize(p-eye_pos);
		// vec3 to_eye = -from_eye;

		vec4 lighting = simplified_lighting(p, rm, eye_pos, ray_dir);
		vec4 col = vec4(rm.material.color, 1.0)*lighting;
		
		//col *= i>1?0.2:1.0;

		#if k_plastic_use
		//color += col_filter*(col * (1.0 - i/1.9))*0.05;
		color += col_filter*(col * (1.0 - i/k_reflection_bounces_max))*k_plastic;
		#endif

		//color += col_filter*col*0.2;

		// color += (vec4(rm.material.emission, 1.0))*col_filter;
		// only use col on the last pass?
		// color += col*col_filter * pow((i+1.0)/(1.0*k_reflection_bounces_max), rm.material.reflection*4.);

		color += col_filter*vec4(rm.material.emission, 1.0);

		// if (i==k_reflection_bounces_max-(rm.material.reflectionr<0.5?20:1)) color += col*col_filter;
		col_filter *= vec4(rm.material.color, 1.0);// *(1.0-rm.material.diffusion*0.8);

		vec3 from_eye_perfect_reflect = reflect(from_eye, norm);
		rm = do_raymarch(p, from_eye_perfect_reflect);
		
		eye_pos = p;
		ray_dir = from_eye_perfect_reflect;

		// make diffuse
		#if k_rnd_diffuse_use
		ray_dir = vec3(rot2(ray_dir.xy, random(vec4(ray_dir, iter_nr*1.0))*k_rnd_diffuse), ray_dir.z);
		#endif

		#if k_fib_diffuse
		// add fibonacci based diffuse?
		iter_count = k_fib_diffuse_count;
		float iter_nr_f = random(vec4(ray_dir.x, i*1.0, iter_nr, seed))*iter_count;
		vec2 golden = vec2(iter_nr_f*(1.0/iter_count), iter_nr_f*golden_ratio_recip);
		golden = rot2(vec2(sqrt(golden.x), 0.), PI2*golden.y); // inside circle of radius 1
		vec3 look_through_p = vec3(golden, 0.);
		vec3 rot_target = look_through_p-vec3(0.,0.,1.);
		vec3 rot_none = vec3(0.)-vec3(0.,0.,1.);
		vec3 rot_angles = rotation_angles_between(rot_none, rot_target);
		ray_dir = rotate_by_angles(ray_dir*0.1, rot_angles);
		#endif
	}

	return color;
}


// main

void main() {

	


	vec2 mp = vec2((mousex-0.5)*2.0*1.4, -2.0*(mousey-0.5)*1.1);

	vec2 frag_coord = gl_FragCoord.xy;
	vec2 raster_size = vec2(window_size_physical_x, window_size_physical_y);

	// x,y as 0..1
	vec2 uv = frag_coord/raster_size.xy;
	// origin at center, x+ -> right, y+ -> up
	uv -= 0.5; uv.y *= -1;
	// correct for aspect ratio
	float aspect_ratio = raster_size.x/raster_size.y;
	uv.x *= aspect_ratio;

	// frag_color = vec4(1.0, 0.0,0.0, 1.0);
	// return;
	
	// uncomment to test hsb
	// float a = angle_of_vec2(uv);
	// // a = rot_angle(a, angle_of_vec2(mp));
	// frag_color = vec4(hsb(a, length(uv)*1.5, 1.0), 1.0);
	// return;

	// frag_color = vec4(floor(uv.x/0.1)*0.1);
	// return;

	// golden ratio demo
	// frag_color = vec4(vec3(0.), 1.);
	// #define n_max 330
	// for (int i = 0; i<n_max; i++) {
	// 	float n = n_max;
	// 	vec2 p = vec2(i*(1.0/n), i*golden_ratio_recip);
	// 	// square grid
	// 	// p.y = p.y - floor(p.y);
	// 	// frag_color.rgb += vec3(1-smoothstep(0.03, 0.04, length(p-uv*2.2)));
	// 	// circular/spiral
	// 	float r = sqrt(p.x);
	// 	float theta = PI2*p.y;
	// 	p = rot2(vec2(r*4.7, 0.), theta);
	// 	frag_color.rgb += vec3(1-smoothstep(0., 0.12, length(p-uv*10.)))*(1-smoothstep(0., n, i*1.));
	// }
	// return;

	
	// camera
	// vec3 eye_pos = vec3(.0+mp.x,.0+mp.y*2.,1.5+mp.y*10.);
	vec3 eye_pos = vec3(.0, .0, 1.5+mp.y*5.);
	// vec3 eye_pos = vec3(.0,.0,5.5);
	float film_offset = 1.0-0.2;
	float film_width = 1.0;
	
	vec2 film_pixel_size = vec2(film_width, film_width) / raster_size.xy; // TODO: aspect_ratio?
	vec3 pixel_center = eye_pos + vec3(uv.xy + film_pixel_size.xy/2.0, -film_offset); // TODO: aspect_ratio?
	
	vec3 ray_dir = normalize(pixel_center - eye_pos);
	// ray_dir.xz = rot2(ray_dir.xz, PI2*mp.x);
	
	RayMarchRes res = do_raymarch(eye_pos, ray_dir);
	
	// float d = box_sdf(vec3(uv, 0.0)-vec3(0.0,mp.y, mp.x), vec3(0.1,0.2,0.1));

	float seed = mousex;
	frag_color = vec4(0.);//trace_color(eye_pos, ray_dir, res);
	for (int i = 0; i<k_iter_count; i++) {
		frag_color += trace_color(eye_pos, ray_dir, res, i, k_iter_count, seed)*(1.0/k_iter_count);
	}
	if (res.dist > k_ray_marching_max_dist) frag_color.x += res.accumulated_closeness;//*0.5;

	// gamma correct
	//float gamma = 2.2;
	float gamma = 2.8;
	frag_color.rgb = pow(frag_color.rgb, vec3(1./gamma));
}

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

const int k_ray_marching_steps = 100;
const float k_ray_marching_start_offset = 0.0001;
const float k_ray_marching_max_dist = 100.0;
const float k_ray_marching_epsilon = 0.0001;
const float k_derivative_epsilon = 0.0001;
const int k_reflection_bounces_max = 2;

const vec2 k_v2 = vec2(1.0,1.0);
const vec3 k_i = vec3(1.,0.,0.);
const vec3 k_j = vec3(0.,1.,0.);
const vec3 k_k = vec3(0.,0.,1.);


// structs

struct Material {
	vec3 color;
};
struct SceneResult {
	float dist;
	Material material;
};


// utils

vec2 rot2(in vec2 v, in float a) {
	// [1, 0] rot PIf2 = [0, 1] = [xc, xs]
	// [0, 1] rot PIf2 = [-1, 0] = [-ys, yc]
	// [x, y] rot a = [(c, -s) (s, c)][x, y] = [xc-ys, xs+yc]
	float s = sin(a);
	float c = cos(a);
	return vec2(v.x*c-v.y*s, v.x*s+v.y*c);
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


// constructors

Material Material_new(in float hue, in float saturation) {
	return Material(hsb(hue, saturation, 1.0));
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

	base = SceneResult(dist, Material(
		mix(base.material.color, added.material.color, percent_added)
	));
}
void union_smooth_sdf(inout SceneResult base, in SceneResult added) {
	union_smooth_sdf(base, added, 0.1);
}



SceneResult scene_sdf_res(in vec3 p) {
	Material white = Material_new(0., 0.);
	SceneResult r = SceneResult(inf, white);
	// r = union_sdf(r, sphere_sdf(p, mousex));
	// r = union_sdf(r, sphere_sdf(p-vec3(.6,0.3,0.2), 0.3));
	// r = union_smooth_sdf(r, sphere_sdf(p-vec3(-.6,0.3,0.2), 0.3), 0.3);

	vec2 mp = vec2((mousex-0.5)*2.0*1.4, -2.0*(mousey-0.5)*1.1);
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
	union_smooth_sdf(r, SceneResult(plane_sdf(p-vec3(1.0,0.0,0.0), normalize(vec3(-1.0, 0.0, 0.0)), 0.), white));
	union_smooth_sdf(r, SceneResult(plane_sdf(p-vec3(0.0,1.0,0.0), normalize(vec3(0.0, -1.0, 0.0)), 0.), white));
	union_smooth_sdf(r, SceneResult(plane_sdf(p-vec3(0.0,0.0,-2.0), normalize(vec3(0.0, 0.0, 1.0)), 0.), white));
	// TODO: make waves radial instead?
	float wave_offset = 0.0;
	wave_offset += sin(p.x*1.1*10.0)*0.01;
	wave_offset += sin(p.z*0.7*10.0)*0.02;
	union_sdf(r, SceneResult(plane_sdf(p-vec3(0.0,-1.0+wave_offset,0.0), normalize(vec3(0.0, 1.0, 0.0)), 0.), Material_new(3.33, 0.5)));

	// camera inside sphere?
	// r = union_smooth_sdf(r, max(sphere_sdf(p- vec3(0.0,0.0,0.0), 3.0), -sphere_sdf(p- vec3(0.0,0.0,0.0), 2.5)), 0.3);

	union_smooth_sdf(r, SceneResult(sphere_sdf(p-vec3(-0.3, -0.3, -0.5), 0.23), Material_new(0., 1.)), 0.2);
	union_smooth_sdf(r, SceneResult(sphere_sdf(p-vec3(0.3, -0.3, -0.3), 0.23), Material_new(3., 1.)), 0.2);
	union_smooth_sdf(r, SceneResult(sphere_sdf(p-vec3(mp.xy, -0.4), 0.23), Material_new(2., 1.)), 0.2);

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

// ray origin, ray direction
vec2 dist_to_surface(in vec3 ro, in vec3 rd) {

	// for antialiasing
	// float first_dist_within_1px = -1.0;
	// float first_dist_within_1px_p = -1.0;

	float closeness_f = 0.0005;
	float acc_closeness = 0;

	float dist = 0;
	float step_dist;
	ro += k_ray_marching_start_offset*rd;
	for (int i=0; i<k_ray_marching_steps; i++) {

		// for antialiasing
		// TODO; also, should be from closest raster pixel edge?
		// float _1px_dist_at_z = 0.001;

		step_dist = scene_sdf(ro);
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
	return vec2(dist, acc_closeness);
}


vec4 trace_color_inner(in vec3 eye_pos, in vec3 ray_dir, float dist) {
	vec3 p = eye_pos + ray_dir*dist;
	vec3 norm = scene_norm(p);

	if (dist >= k_ray_marching_max_dist) {
		// sky/background
		// float c = (uv.y+0.5+0.2)*0.04;
		// frag_color = vec4(0.0,c*0.2,c, 1.0);
		return vec4(0.0,0.0,0.0,0.0);
	}

	float ambient_lighting = 0.0001;

	float c = 0;
	c += ambient_lighting;


	vec3 from_eye = normalize(p-eye_pos);
	vec3 to_eye = -from_eye;

	{
		// vec3 light_pos = vec3(0.3, 1.3,(mousex-0.5)*3.0);
		vec3 light_pos = vec3(0.0, 0.0, 0.0);
		// vec3 light_pos = vec3((mousex-0.5)*2.0, -2.0*(mousey-0.5), 0.0);
		float intensity = 0.3;

		vec3 from_light = normalize(p-light_pos);
		vec3 perfect_reflection = reflect(from_light, norm);

		float cos_angle_to_perfect = dot(perfect_reflection, norm);
		float angle_to_perfect_10 = 1-min(1, (acos(cos_angle_to_perfect)/PIf2));
		c += angle_to_perfect_10*intensity;

		float specular_intensity = 0.005/pow(mousex*10.0, 10.0);
		float specular_shininess = 5.5*mousex*10.0;

		float cos_angle_eye_to_reflection = dot(perfect_reflection, from_eye);
		float angle_eye_to_reflection_10 = acos(cos_angle_eye_to_reflection)/PIf2;
		c += pow(angle_eye_to_reflection_10, specular_shininess)*specular_intensity;
	}

	// vec4 color = vec4(c, c, c, 1.0);
	vec3 col = scene_color(p);
	vec4 color = vec4(col.x*c, col.y*c, col.z*c, 1.0);

	return color;
	// frag_color = vec4(norm, 1.0);
}

vec4 trace_color(in vec3 eye_pos, in vec3 ray_dir, float dist) {

	vec4 color = vec4(0.0,0.0,0.0,0.0);

	for (int i = 0; i<k_reflection_bounces_max; i++) {
		vec3 p = eye_pos + ray_dir*dist;
		vec3 norm = scene_norm(p);
		vec3 from_eye = normalize(p-eye_pos);
		vec3 to_eye = -from_eye;

		color += trace_color_inner(eye_pos, ray_dir, dist) * (1.0 - i/1.9);

		vec3 from_eye_perfect_reflect = reflect(from_eye, norm);
		vec2 reflection_dist2 = dist_to_surface(p, from_eye_perfect_reflect);

		eye_pos = p;
		ray_dir = from_eye_perfect_reflect;
		dist = reflection_dist2.x;
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

	// uncomment to test hsb
	// float a = angle_of_vec2(uv);
	// // a = rot_angle(a, angle_of_vec2(mp));
	// frag_color = vec4(hsb(a, length(uv)*1.5, 1.0), 1.0);
	// return;


	// camera
	vec3 eye_pos = vec3(.0,.0,1.5);
	float film_offset = 1.0;
	float film_width = 1.0;

	vec2 film_pixel_size = vec2(film_width, film_width) / raster_size.xy; // TODO: aspect_ratio?
	vec3 pixel_center = eye_pos + vec3(uv.xy + film_pixel_size.xy/2.0, -film_offset); // TODO: aspect_ratio?

	vec3 ray_dir = normalize(pixel_center - eye_pos);
	vec2 dist2 = dist_to_surface(eye_pos, ray_dir);
	float dist = dist2.x;
	float acc_closeness = dist2.y;

	// float d = box_sdf(vec3(uv, 0.0)-vec3(0.0,mp.y, mp.x), vec3(0.1,0.2,0.1));
	// frag_color = vec4(d, d, d, 1.0);
	// return;

	frag_color = trace_color(eye_pos, ray_dir, dist);
	if (dist > k_ray_marching_max_dist) frag_color.x += acc_closeness;//*0.5;
}

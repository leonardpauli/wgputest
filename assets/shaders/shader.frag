#version 450

in vec4 gl_FragCoord;
layout(location = 0) out vec4 frag_color;
layout(set = 0, binding = 0) uniform Uniforms {
	float mousex;
};

// TODO: gamma correct
// TODO: antialiasing

// some inspired by http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/


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
const vec2 k_v2 = vec2(1.0,1.0);


// all sdf's are from origin?

float sphere_sdf(in vec3 p, in float r) {
	return length(p) - r;
}

float union_sdf(in float a, in float b) {
	return min(a, b);
}

// try k=0.1
float union_smooth_sdf(in float a, in float b, in float k) {
	float h = max(0, k-abs(a-b))/k;
	return min(a, b) - h*h*k*(1.0/4.0); // mult by recip faster
}

float scene_sdf(in vec3 p) {
	float r = inf;
	r = union_sdf(r, sphere_sdf(p, mousex));
	r = union_sdf(r, sphere_sdf(p-vec3(.6,0.3,0.2), 0.3));
	r = union_smooth_sdf(r, sphere_sdf(p-vec3(-.6,0.3,0.2), 0.3), 0.3);
	// r = union_sdf(r, sphere_sdf(p-xy*0.5, 0.1));
	// r = union_sdf(r, sphere_sdf(p-xy*0.5, 0.1));
	// r = union_sdf(r, sphere_sdf(p-vec3(0.5), 0.1));
	return r;
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


vec4 trace_color(in vec3 eye_pos, in vec3 ray_dir, float dist) {
	vec3 p = eye_pos + ray_dir*dist;
	vec3 norm = scene_norm(p);

	if (dist >= k_ray_marching_max_dist) {
		// sky/background
		// float c = (uv.y+0.5+0.2)*0.04;
		// frag_color = vec4(0.0,c*0.2,c, 1.0);
		return vec4(0.0,0.0,0.0,0.0);
	}

	float ambient_lighting = 0.004;

	float c = 0;
	c += ambient_lighting;


	{
		vec3 light_pos = vec3(0.3, 1.3,(mousex-0.5)*3.0);
		float intensity = 0.5;

		vec3 from_light = normalize(p-light_pos);
		vec3 perfect_reflection = reflect(from_light, norm);

		float cos_angle_to_perfect = dot(perfect_reflection, norm);
		float angle_to_perfect_10 = 1-min(1, (acos(cos_angle_to_perfect)/PIf2));
		c += angle_to_perfect_10*intensity;

		float specular_intensity = 0.005;
		float specular_shininess = 5.5;

		vec3 to_eye = normalize(p-eye_pos);
		float cos_angle_eye_to_reflection = dot(perfect_reflection, to_eye);
		float angle_eye_to_reflection_10 = acos(cos_angle_eye_to_reflection)/PIf2;
		c += pow(angle_eye_to_reflection_10, specular_shininess)*specular_intensity;
	}

	return vec4(c, c, c, 1.0);
	// frag_color = vec4(norm, 1.0);
}


// main

void main() {
	vec2 frag_coord = gl_FragCoord.xy;
	vec2 raster_size = vec2(1400.0, 1000.0);

	// x,y as 0..1
	vec2 uv = frag_coord/raster_size.xy;
	// origin at center, x+ -> right, y+ -> up
	uv -= 0.5; uv.y *= -1;
	// correct for aspect ratio
	float aspect_ratio = raster_size.x/raster_size.y;
	uv.x *= aspect_ratio;

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

	frag_color = trace_color(eye_pos, ray_dir, dist);
	if (dist > k_ray_marching_max_dist) frag_color.x += acc_closeness*0.05;
}

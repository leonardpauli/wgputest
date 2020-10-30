#version 450

in vec4 gl_FragCoord;
layout(location = 0) out vec4 frag_color;


// consts

const float inf = 1.0/0.0;


// all sdf's are from origin?

float sphere_sdf(in vec3 p, in float r) {
	return length(p) - r;
}

float union_sdf(in float a, in float b) {
	return min(a, b);
}

float scene_sdf(in vec3 p) {
	float r = inf;
	r = union_sdf(r, sphere_sdf(p, 0.5));
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

	float k_derivative_epsilon = 0.000001;
	float h = k_derivative_epsilon;
	float o = scene_sdf(p);
	vec3 grad = vec3(
		scene_sdf(p+vec3(h,.0,.0)),
		scene_sdf(p+vec3(.0,h,.0)),
		scene_sdf(p+vec3(.0,.0,h))
	) - o;

	return normalize(grad);
}

// ray origin, ray direction
float dist_to_surface(in vec3 ro, in vec3 rd) {
	const int k_ray_marching_steps = 100;
	const float k_ray_marching_start_offset = 0.0001;
	const float k_ray_marching_max_dist = 100.0;
	const float k_ray_marching_epsilon = 0.0001;

	float dist = 0;
	ro += k_ray_marching_start_offset*rd;
	for (int i=0; i<k_ray_marching_steps; i++) {
		float step_dist = scene_sdf(ro);
		dist += step_dist;
		if (step_dist <= k_ray_marching_epsilon) return dist;
		if (dist >= k_ray_marching_max_dist) return dist;
		ro += step_dist*rd;
	}
	return dist;
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

	// TODO: antialiasing; use pixel_width = ...; smooth_step(x, x+pixel_width...)


	// camera
	vec3 eye_pos = vec3(.0,.0,1.5);
	float film_offset = 1.0;
	float film_width = 1.0;

	vec2 film_pixel_size = vec2(film_width, film_width) / raster_size.xy; // TODO: aspect_ratio?
	vec3 pixel_center = eye_pos + vec3(uv.xy + film_pixel_size/2.0, -film_offset); // TODO: aspect_ratio?
	vec3 ray_dir = normalize(pixel_center - eye_pos);

	float dist = dist_to_surface(eye_pos, ray_dir);
	vec3 p = eye_pos + ray_dir*dist;
	vec3 norm = scene_norm(p);

	vec3 light_pos = vec3(1.0, 1.0, 0.0);
	vec3 to_light = normalize(light_pos-p);
	vec3 perfect_reflection = dot(to_light, norm)-to_light;
	float angle_to_perfect = acos(dot(perfect_reflection, norm));

	float c = angle_to_perfect / (3.14159/2.0);
	frag_color = vec4(c, c, c, 1.0);
}

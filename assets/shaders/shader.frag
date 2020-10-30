#version 450

in vec4 gl_FragCoord;
layout(location = 0) out vec4 outColor;

// void main() {
// 	vec2 ires = vec2(100.0, 100.0);
// 	vec2 uv = gl_FragCoord.xy/ires - vec2(1.0, 1.0)*10.0;

// 	float d = length(uv) - 2.7;

// 	outColor = vec4(0.9, d, d, 1.0);
// }


float scene_sdf(in vec3 p) {
	return length(p) - 0.3;
}
vec3 scene_norm(in vec3 p) {
    // normal is perpendicular to tangent
    // tangent's tilt is the derivative
    // derivative is dy/dx
    // dy/dx is lim(h->0) of f(a+h)-f(a) / h
    // in multi-variable, derivative is called gradient

    float h = 0.001;
    float o = scene_sdf(p);
    vec3 grad = vec3(
    	scene_sdf(p+vec3(h,.0,.0)),
		scene_sdf(p+vec3(.0,h,.0)),
		scene_sdf(p+vec3(.0,.0,h))
    ) - o;

	return grad;
}

void main() {
	vec2 fragCoord = gl_FragCoord.xy;
	vec2 iResolution = vec2(1000.0, 1000.0);
	float epsilon = 1.0/max(iResolution.x, iResolution.y)*3.0;
	vec3 v1 = vec3(1.0,1.0,1.0);
	vec3 v2 = v1/2.0;

	vec2 uv = fragCoord/iResolution.xy-v2.xy;
	uv /= 0.99;
	float w = iResolution.x/iResolution.y;
	uv.x *= w;

	// outColor = vec4(uv.xy, 0.0, 1.0);
	// return;

	// ray origin, ray direction; ray marching SDF
	// ray from camera goes from +z -> -z
	float cam_off = 2000.0;
	float film_off = 1.0;
	vec3 co = vec3(0.0,0.0,film_off+cam_off);
	vec3 ro = vec3(uv.xy, film_off);
	vec3 rd = normalize(ro-co);

	// step in dir
	vec3 p = ro;
	float acc_dist = 0.0;

	int MAX_STEPS = 5;
	int i=0;
	for (; i<MAX_STEPS; i++) {
		float mindist = scene_sdf(p);
			if (mindist<0.0) break;
			if (mindist<epsilon) break;
		p = ro+rd*mindist;
		acc_dist += mindist;
	}
	if (i>=MAX_STEPS) {
		acc_dist = 1.0/0.0;
	}


	float c = acc_dist*0.1;
	c = c>10000.0?0.0:1.0;
	vec3 grad = normalize(scene_norm(p));

	outColor = vec4(grad.xyz*c,1.0);
}

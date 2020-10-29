#version 450

in vec4 gl_FragCoord;
layout(location = 0) out vec4 outColor;

void main() {
	vec2 ires = vec2(100.0, 100.0);
	vec2 uv = gl_FragCoord.xy/ires;

	float d = length(uv) - 2.7;

	outColor = vec4(0.7, d, d, 1.0);
}

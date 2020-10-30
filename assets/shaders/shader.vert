#version 450

out gl_PerVertex {
	vec4 gl_Position;
};

void main() {
	vec2[] poss = {
		vec2(-1.0, -1.0),
		vec2(-1.0, 1.0),
		vec2(1.0, 1.0),
		vec2(1.0, -1.0),
		vec2(-1.0, -1.0),
		vec2(1.0, 1.0)
	};
	vec2 pos = poss[gl_VertexIndex];
	gl_Position = vec4(pos, 0.0, 1.0);
}

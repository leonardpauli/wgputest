#version 450

in vec4 gl_FragCoord;
layout(location = 0) out vec4 frag_color;

void main() {
	vec2 frag_coord = gl_FragCoord.xy;
	vec2 raster_size = vec2(1400.0, 1000.0);

	// x,y as 0..1
	vec2 uv = frag_coord/raster_size.xy;
	// origin at center, x+ -> right, y+ -> up
	uv -= 0.5; uv.y *= -1;
	// correct for aspect ratio
	uv.x *= raster_size.x/raster_size.y;

	frag_color = vec4(uv.xy, 0.0, 1.0);
}

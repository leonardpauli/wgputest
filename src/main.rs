// starting point inspired from wgpu/examples/hello-triangle

use bytemuck;
use winit::{
	event::{Event, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
	window::Window,
};
use wgpu::util::{DeviceExt};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};


use std::io::Read;
use std::path::{Path, PathBuf};
// use std::io::prelude::*;


const WINDOW_SIZE_PHYSICAL: (u32, u32) = (1400/2, 1000/2);


fn main() {
	
	let parent = std::env::current_exe().unwrap();
	let parent = parent.parent().unwrap();
	let assets_dir = if parent.file_name().unwrap().to_str().unwrap() == "MacOS" {
		parent.parent().unwrap().join(Path::new("Resources")).join(Path::new("assets"))
	} else {
		// std::path::PathBuf::from("assets")
		std::env::current_dir().unwrap().join(Path::new("assets"))
	};

	let event_loop = EventLoop::new();
	let window = winit::window::WindowBuilder::new()
		.with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_SIZE_PHYSICAL.0, WINDOW_SIZE_PHYSICAL.1))
		.with_title("wgputest by Leonard Pauli, oct 2020")
		.build(&event_loop).unwrap();
	// #[cfg(not(target_arch = "wasm32"))]
	{
		// subscriber::initialize_default_subscriber(None);
		// Temporarily avoid srgb formats for the swapchain on the web
		futures::executor::block_on(run(event_loop, window, wgpu::TextureFormat::Bgra8UnormSrgb, &assets_dir));
	}
	// #[cfg(target_arch = "wasm32")]
	// {
	// 		std::panic::set_hook(Box::new(console_error_panic_hook::hook));
	// 		console_log::init().expect("could not initialize logger");
	// 		use winit::platform::web::WindowExtWebSys;
	// 		// On wasm, append the canvas to the document body
	// 		web_sys::window()
	// 				.and_then(|win| win.document())
	// 				.and_then(|doc| doc.body())
	// 				.and_then(|body| {
	// 						body.append_child(&web_sys::Element::from(window.canvas()))
	// 								.ok()
	// 				})
	// 				.expect("couldn't append canvas to document body");
	// 		wasm_bindgen_futures::spawn_local(run(event_loop, window, wgpu::TextureFormat::Bgra8Unorm));
	// }
}

fn load_shader_module<'a, P: AsRef<std::path::Path>>(
	device: &wgpu::Device,
	p: P,
) -> wgpu::ShaderModule {
	// TODO: handle errors
	// let a: &std::path::Path = p.as_ref();
	// println!("load_shader_module: {:?}", a.canonicalize());
	let file = std::fs::File::open(p);
	let mut buf = Vec::new();
	file.unwrap().read_to_end(&mut buf).unwrap();
	let res = wgpu::util::make_spirv(&buf[0..]);
	device.create_shader_module(res)
}

#[repr(C)] // make compatible with shader
#[derive(Debug, Copy, Clone)] // make storable in buffer
struct Uniforms {
	window_size_physical_x: u32,
	window_size_physical_y: u32,
	mousex: f32,
	mousey: f32,
}
unsafe impl bytemuck::Pod for Uniforms {} // ?
unsafe impl bytemuck::Zeroable for Uniforms {} // ?
impl Uniforms {
	fn new() -> Self {
		Self {
			window_size_physical_x: WINDOW_SIZE_PHYSICAL.0,
			window_size_physical_y: WINDOW_SIZE_PHYSICAL.1,
			mousex: 0.5,
			mousey: 0.5,
		}
	}
}

async fn run(event_loop: EventLoop<()>, window: Window, swapchain_format: wgpu::TextureFormat, assets_dir: &PathBuf) {
	let size = window.inner_size();
	let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
	let surface = unsafe { instance.create_surface(&window) };

	let adapter = instance
		.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::default(),
			compatible_surface: Some(&surface),
		})
		.await
		.expect("Failed to find an appropiate adapter");

	let (device, queue) = adapter
		.request_device(
			&wgpu::DeviceDescriptor {
				features: wgpu::Features::empty(),
				limits: wgpu::Limits::default(),
				shader_validation: true,
			},
			None,
		)
		.await
		.expect("Failed to create device");
	let device: wgpu::Device = device;

	let frag_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
		label: Some("frag_bind_group_layout"),
		entries: &[
			wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStage::FRAGMENT,
				ty: wgpu::BindingType::UniformBuffer {
					dynamic: false,
					min_binding_size: None, // wgpu::BufferSize::new(std::mem::size_of::<f64>() as _),
				},
				count: None,
			},
		],
	});

	let mut frag_uniforms = Uniforms::new();
	let frag_uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("frag_uniforms"),
		contents: bytemuck::cast_slice(&[frag_uniforms]),
		usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
	});

	let frag_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		label: Some("frag_bind_group"),
		layout: &frag_bind_group_layout,
		entries: &[
			wgpu::BindGroupEntry {
				binding: 0,
				resource: wgpu::BindingResource::Buffer(frag_uniforms_buf.slice(..)),
			},
		],
	});

	let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
		label: None,
		bind_group_layouts: &[
			&frag_bind_group_layout,
		],
		push_constant_ranges: &[],
	});

	let mut sc_desc = wgpu::SwapChainDescriptor {
		usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
		format: swapchain_format,
		width: size.width,
		height: size.height,
		present_mode: wgpu::PresentMode::Mailbox,
	};

	let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

	let wr = std::sync::Arc::from(window);
	let window = wr.clone();

	let shaders_source_dir = assets_dir.join(Path::new("shaders"));
	let asset_gen_spv_dir = assets_dir.join(Path::new("gen/spv"));

	let shaders_source_dir2 = shaders_source_dir.clone();
	fn compile_shader_at_path(dir: &PathBuf, p: &std::path::PathBuf) {
		let name = p.components().last().unwrap();
		let source = p.to_str().unwrap();
		let target = dir.join(name);

		let start = std::time::Instant::now();
		println!("shader.compile.start({:?})", name);
		let res = std::process::Command::new("glslangValidator")
			.arg(source)
			.arg("-V")
			.arg("-o").arg(target)
			.output()
			.expect("failed to execute process");

		println!("shader.compile.end({:?}) {}ms status({})", name, start.elapsed().as_millis(), res.status);
		if !res.status.success() {
			println!(
				"\t---{{\nexec out: {}\nerr: {};\n\t}}---",
				String::from_utf8_lossy(&res.stdout[0..]),
				String::from_utf8_lossy(&res.stderr[0..])
			);
		}
	}

	if !asset_gen_spv_dir.is_dir() {
		std::fs::create_dir_all(&asset_gen_spv_dir).unwrap();

		for entry in std::fs::read_dir(shaders_source_dir2).unwrap().map(|x| x.unwrap()) {
			let entry: std::fs::DirEntry = entry;
			let ignore = entry.file_name().to_str().unwrap().starts_with(".");
			if ignore {continue;}
			compile_shader_at_path(&asset_gen_spv_dir, &entry.path());
		}

	}

	let render_pipeline_needs_update = std::sync::Mutex::new(true);
	let render_pipeline_needs_update1 = std::sync::Arc::new(render_pipeline_needs_update);
	let render_pipeline_needs_update2 = std::sync::Arc::clone(&render_pipeline_needs_update1);

	let asset_gen_spv_dir2 = asset_gen_spv_dir.clone();
	let mut watcher: RecommendedWatcher = Watcher::new_immediate(move |res| match res {
		Ok(event) => {
			let event: notify::Event = event;

			match event.kind {
				notify::EventKind::Modify(notify::event::ModifyKind::Data(_))=> {
					// TODO: debouncing? sometimes getting two consecutive
					// println!("event: {:?}", event);
					for p in event.paths.iter() {
						let p: &std::path::PathBuf = p;
						let p = p.canonicalize().unwrap();
						compile_shader_at_path(&asset_gen_spv_dir2, &p);
					}
					*render_pipeline_needs_update1.lock().unwrap() = true;
					wr.request_redraw();
				},
				_=> {}, // println!("event: {:?}", event)
			}
		}
		Err(e) => println!("watch error: {:?}", e),
	})
	.unwrap();
	watcher
		.watch(shaders_source_dir, RecursiveMode::Recursive)
		.unwrap();

	let mut vs_module: Option<wgpu::ShaderModule> = None;
	let mut fs_module: Option<wgpu::ShaderModule> = None;
	let mut render_pipeline: Option<wgpu::RenderPipeline> = None;

	let asset_gen_spv_dir = asset_gen_spv_dir.clone();
	event_loop.run(move |event, _, control_flow| {
		// println!("{:?}", event);

		// Have the closure take ownership of the resources.
		// `event_loop.run` never returns, therefore we must do this to ensure
		// the resources are properly cleaned up.
		let _ = (
			&instance,
			&adapter,
			&vs_module,
			&fs_module,
			&pipeline_layout,
			&frag_bind_group,
		);

		// let timer_dur = Duration::from_millis(300);

		/*
		WindowEvent {
			window_id: WindowId(Id(140254236424576)),
			event: KeyboardInput {
				device_id: DeviceId(DeviceId),
				input: KeyboardInput {
					scancode: 15, state: Released,
					virtual_keycode: Some(R),
					modifiers: (empty)
				}, is_synthetic: false }
			}
			*/

		let mut render_pipeline_needs_update = render_pipeline_needs_update2.lock().unwrap();
		if *render_pipeline_needs_update {
			*render_pipeline_needs_update = false;

			let start = std::time::Instant::now();
			vs_module = Some(load_shader_module(&device, asset_gen_spv_dir.join("shader.vert")));
			fs_module = Some(load_shader_module(&device, asset_gen_spv_dir.join("shader.frag")));
			println!("shader.load x2 ({}ms)", start.elapsed().as_millis());

			let start = std::time::Instant::now();
			render_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
				label: None,
				layout: Some(&pipeline_layout),
				vertex_stage: wgpu::ProgrammableStageDescriptor {
					module: &vs_module.as_ref().unwrap(),
					entry_point: "main",
				},
				fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
					module: &fs_module.as_ref().unwrap(),
					entry_point: "main",
				}),
				// Use the default rasterizer state: no culling, no depth bias
				rasterization_state: None,
				primitive_topology: wgpu::PrimitiveTopology::TriangleList,
				color_states: &[swapchain_format.into()],
				depth_stencil_state: None,
				vertex_state: wgpu::VertexStateDescriptor {
					index_format: wgpu::IndexFormat::Uint16,
					vertex_buffers: &[],
				},
				sample_count: 1,
				sample_mask: !0,
				alpha_to_coverage_enabled: false,
			}));

			println!("render_pipeline.created ({}ms)", start.elapsed().as_millis());
		}

		*control_flow = ControlFlow::WaitUntil(std::time::Instant::now() + std::time::Duration::from_millis(100));
		match event {
			Event::WindowEvent {
				event:
					WindowEvent::KeyboardInput {
						input:
							winit::event::KeyboardInput {
								virtual_keycode: Some(winit::event::VirtualKeyCode::R),
								..
							},
						..
					},
				..
			} => {
				window.request_redraw();
			}
			// Event::NewEvents(StartCause::Init)=> {
			// 	*control_flow = ControlFlow::WaitUntil(Instant::now() + timer_dur);
			// },
			// Event::NewEvents(StartCause::ResumeTimeReached {..})=> {
			// 	*control_flow = ControlFlow::WaitUntil(Instant::now() + timer_dur);
			// 	println!("Time!");
			// },
			Event::WindowEvent {
				event: WindowEvent::CursorMoved { position, .. },
				..
			}=> {
				let position: winit::dpi::PhysicalPosition<_> = position;

				frag_uniforms.mousex = (position.x as f32)/(frag_uniforms.window_size_physical_x as f32);
				frag_uniforms.mousey = (position.y as f32)/(frag_uniforms.window_size_physical_y as f32);
				queue.write_buffer(&frag_uniforms_buf, 0, bytemuck::cast_slice(&[frag_uniforms]));
				// window.request_redraw();

				// println!("{:?}", position.x);
			},
			Event::WindowEvent {
				event: WindowEvent::Resized(size),
				..
			} => {
				// Recreate the swap chain with the new size

				frag_uniforms.window_size_physical_x = size.width;
				frag_uniforms.window_size_physical_y = size.height;

				sc_desc.width = size.width;
				sc_desc.height = size.height;
				swap_chain = device.create_swap_chain(&surface, &sc_desc);
				// window.request_redraw();
			}
			Event::RedrawRequested(_) => {
				let frame = swap_chain
					.get_current_frame()
					.expect("Failed to acquire next swap chain texture")
					.output;

				let mut encoder =
					device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
				{
					let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
						color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
							attachment: &frame.view,
							resolve_target: None,
							ops: wgpu::Operations {
								load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
								store: true,
							},
						}],
						depth_stencil_attachment: None,
					});
					rpass.set_pipeline(&render_pipeline.as_ref().unwrap());
					rpass.set_bind_group(0, &frag_bind_group, &[]);
					rpass.draw(0..6, 0..1);
				}

				queue.submit(Some(encoder.finish()));
			}
			Event::WindowEvent {
				event: WindowEvent::CloseRequested,
				..
			} => *control_flow = ControlFlow::Exit,
			_ => {}
		}
	});
}

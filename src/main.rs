// starting point inspired from wgpu/examples/hello-triangle

// TODO: read all vscode hover doc boxes
// TODO: online/built-in compilation of shader.*

use winit::{
	event::{Event, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
	window::Window,
};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};


use std::io::Read;

fn main() {
	let event_loop = EventLoop::new();
	let window = winit::window::WindowBuilder::new()
		.with_inner_size(winit::dpi::LogicalSize::new(1400/2, 1000/2))
		.build(&event_loop).unwrap();
	// #[cfg(not(target_arch = "wasm32"))]
	{
		// subscriber::initialize_default_subscriber(None);
		// Temporarily avoid srgb formats for the swapchain on the web
		futures::executor::block_on(run(event_loop, window, wgpu::TextureFormat::Bgra8UnormSrgb));
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

async fn run(event_loop: EventLoop<()>, window: Window, swapchain_format: wgpu::TextureFormat) {
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

	let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
		label: None,
		bind_group_layouts: &[],
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


	const ASSET_GEN_SPV_DIR_STR: &str = "assets/gen/spv";
	const SHADER_SRC_DIR_STR: &str = "assets/shaders";
	let shaders_source_dir = std::path::Path::new(SHADER_SRC_DIR_STR);
	let asset_gen_spv_dir = std::path::Path::new(ASSET_GEN_SPV_DIR_STR);

	fn compile_shader_at_path(p: &std::path::PathBuf) {
		let dir = std::path::Path::new(ASSET_GEN_SPV_DIR_STR);
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
		let dir = asset_gen_spv_dir;
		std::fs::create_dir_all(dir).unwrap();

		for entry in std::fs::read_dir(shaders_source_dir).unwrap().map(|x| x.unwrap()) {
			let entry: std::fs::DirEntry = entry;
			let ignore = entry.file_name().to_str().unwrap().starts_with(".");
			if ignore {continue;}
			compile_shader_at_path(&entry.path());
		}

	}

	let render_pipeline_needs_update = std::sync::Mutex::new(true);
	let render_pipeline_needs_update1 = std::sync::Arc::new(render_pipeline_needs_update);
	let render_pipeline_needs_update2 = std::sync::Arc::clone(&render_pipeline_needs_update1);

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
						compile_shader_at_path(&p);
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
			vs_module = Some(load_shader_module(&device, "assets/gen/spv/shader.vert"));
			fs_module = Some(load_shader_module(&device, "assets/gen/spv/shader.frag"));
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
				event: WindowEvent::Resized(size),
				..
			} => {
				// Recreate the swap chain with the new size
				sc_desc.width = size.width;
				sc_desc.height = size.height;
				swap_chain = device.create_swap_chain(&surface, &sc_desc);
				window.request_redraw();
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

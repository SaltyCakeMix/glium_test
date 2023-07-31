#[macro_use]
extern crate glium;
extern crate image;

use std::path::Path;

// Vertex for OpenGL
#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}
implement_vertex!(Vertex, position, tex_coords, normal);

fn load_obj<P>(path: P) -> std::io::Result<(Vec<Vertex>, Vec<u16>)>
where P: AsRef<Path> {
    use std::io::{BufReader, BufRead};
    use std::collections::HashSet;

    let file = std::fs::File::open(path)?;
    let lines: Vec<String> = BufReader::new(file).lines().collect::<Result<_, _>>().unwrap();

    let mut v: Vec<Vertex> = Vec::new();
    let mut vt: Vec<[f32; 2]> = Vec::new();
    let mut vn: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    let mut set: HashSet<u16> = HashSet::new(); 

    fn split_read(split: &mut std::str::Split<&str>) -> f32 {
        split.next().unwrap().parse::<f32>().unwrap()
    }
    fn parse_index(split: &mut std::str::Split<&str>) -> usize {
        (split_read(split) - 1.0) as usize
    }

    for line in &lines {
        let mut split = line.split(" ");
        match split.next().unwrap() {
            "v" => v.push( Vertex {
                position:
                [   split_read(&mut split) + 2.5,
                    split_read(&mut split),
                    split_read(&mut split),
                ],
                tex_coords: [0.0; 2],
                normal: [0.0; 3],
            }),
            "vt" => vt.push(
                [   split_read(&mut split),
                    split_read(&mut split),
                ]),
            "vn" => vn.push(
                [   split_read(&mut split),
                    split_read(&mut split),
                    split_read(&mut split),
                ]),
            "f" => {
                // Stores corner indices for polygon
                let mut corners: Vec<u16> = Vec::new();

                // Finds corner indices
                for corner in split {
                    let mut iter = corner.split("/");
                    let index = parse_index(&mut iter);
                    corners.push(index as u16);

                    // Updates Vertex if it hasn't come across it yet
                    if set.insert(index as u16) {
                        let vertex = &mut v[index];
                        vertex.tex_coords = vt[parse_index(&mut iter)];
                        vertex.normal = vn[parse_index(&mut iter)];
                    }
                }
                
                // Creates triangles for polygon
                for i in 1..corners.len()-1 {
                    indices.push(corners[0]);
                    indices.push(corners[i]);
                    indices.push(corners[i + 1]);
                }
            },
            _ => (),
        }
    }

    Ok((v, indices))
}

fn main() {
    use glium::{glutin, Surface};

    // Creates display
    let events_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_inner_size(glutin::dpi::LogicalSize::new(800.0, 800.0))
        .with_title("Hello world");
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &events_loop).unwrap();

    // Shaders
    let vertex_shader_src = r#"
        #version 150

        in vec3 position;
        in vec3 normal;
        in vec2 tex_coords;

        out vec3 v_normal;
        out vec3 v_position;
        out vec2 v_tex_coords;

        uniform mat4 perspective;
        uniform mat4 view;
        uniform mat4 model;

        void main() {
            v_tex_coords = tex_coords;
            mat4 modelview = view * model;
            v_normal = transpose(inverse(mat3(modelview))) * normal;
            gl_Position = perspective * modelview * vec4(position, 1.0);
            v_position = gl_Position.xyz / gl_Position.w;
        }
    "#;

    let fragment_shader_src = r#"
        #version 140
    
        in vec3 v_position;
        in vec3 v_normal;
        in vec2 v_tex_coords;

        out vec4 color;

        uniform vec3 u_light;
        uniform sampler2D diffuse_tex;

        const vec3 specular_color = vec3(1.0, 1.0, 1.0);

        void main() {
            vec3 diffuse_color = texture(diffuse_tex, v_tex_coords).rgb;
            vec3 ambient_color = diffuse_color * 0.1;

            float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);

            vec3 camera_dir = normalize(-v_position);
            vec3 half_direction = normalize(normalize(u_light) + camera_dir);
            float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);

            color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
        }
    "#;
    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let Ok((s, i)) = load_obj("fish.obj") else {panic!();};
    let shape = glium::VertexBuffer::new(&display, &s).unwrap();
    let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &i).unwrap();

    // Loads image
    let image = image::open("fish_texture.png").unwrap().to_rgba8();
    let image_dimensions = image.dimensions();
    let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
    let diffuse_texture = glium::texture::SrgbTexture2d::new(&display, image).unwrap();

    let mut t: f32 = 0.0;
    let mut k: f32 = 0.0;

    // Run
    events_loop.run(move |ev, _, control_flow| {
        
        // Update
        let mut target = display.draw();

        t += 0.005;
        if t > std::f32::consts::TAU {
            t = 0.0;
        }
        k += 0.01;
        if k > 3.0 {
            k = -2.5;
        }

        let perspective = {
            let (width, height) = target.get_dimensions();
            let aspect_ratio = height as f32 / width as f32;

            let fov: f32 = 3.141592 / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;

            let f = 1.0 / (fov / 2.0).tan();

            [
                [f * aspect_ratio, 0.0,              0.0              , 0.0],
                [       0.0      ,   f,              0.0              , 0.0],
                [       0.0      , 0.0,  (zfar+znear)/(zfar-znear)    , 1.0],
                [       0.0      , 0.0, -(2.0*zfar*znear)/(zfar-znear), 0.0],
            ]
        };
        let view = view_matrix(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[0.0, 1.0, 0.0],
        );
        let uniforms = uniform! {
            model: [
                [0.5 * t.cos(), 0.0, 0.5 * -t.sin(), 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.5 * t.sin(), 0.0, 0.5 * t.cos(), 0.0],
                [0.0, k, 3.0, 1.0f32],
            ],
            view:view,
            perspective: perspective,
            u_light: [-1.0, 0.4, 0.9f32],
            diffuse_tex: &diffuse_texture,
        };
        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            //backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
            .. Default::default()
        };

        // Draw
        target.clear_color_and_depth((0.0, 0.0, 1.0, 1.0), 1.0);
        target.draw(&shape, &indices, &program, &uniforms, &params).unwrap();
        target.finish().unwrap();

        // Window events
        match ev {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                }
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => (),
        }

        // Wait till next frame
        let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);
    });
}

fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let f = direction;
        let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let len = len.sqrt();
        [f[0] / len, f[1] / len, f[2] / len]
    };

    let s = [up[1] * f[2] - up[2] * f[1],
             up[2] * f[0] - up[0] * f[2],
             up[0] * f[1] - up[1] * f[0]];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [f[1] * s_norm[2] - f[2] * s_norm[1],
             f[2] * s_norm[0] - f[0] * s_norm[2],
             f[0] * s_norm[1] - f[1] * s_norm[0]];

    let p = [-position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
             -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
             -position[0] * f[0] - position[1] * f[1] - position[2] * f[2]];

    [
        [s_norm[0], u[0], f[0], 0.0],
        [s_norm[1], u[1], f[1], 0.0],
        [s_norm[2], u[2], f[2], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}
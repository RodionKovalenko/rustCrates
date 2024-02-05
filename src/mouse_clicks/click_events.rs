use enigo::*;
use std::process::Command;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use webbrowser;
use rand::Rng;

pub fn simulate_click_events() {
    let mut enigo = Enigo::new();
    let mut now = Instant::now();
    let number_of_accounts:i32 = 7;
    let search_items = ["backpropagation alternatives", "how to get much money",
        "it is possible to get access to browser api?", "be healthy", "are fruits healthy", "what is our future?", "verschwörungstheorie existiert?",
        "recurrent neural network are still good?", "how to get a million dollar?", "statistics forever?", "why are we getting older?",
        "can we live forever?", "brave is the best browser?", "who am I?", "is it good to be faithful to your partner?",
        "why not give people enough money?"
    ];
    let mut rng = rand::thread_rng();
    let time = now.elapsed();
    println!("{:?}", time);

    loop {
        println!("time elapsed {:?}", time);

        if webbrowser::open("https://www.google.de/").is_ok() {
            thread::sleep(Duration::from_secs(2));
            enigo.key_click(Key::Return);

            enigo.key_sequence_parse(search_items[rng.gen_range(0, search_items.len())]);

            enigo.key_click(Key::Return);
            thread::sleep(Duration::from_secs(5));
            enigo.key_down(Key::Control);
            enigo.key_down(Key::F4);
            enigo.key_up(Key::Control);
            enigo.key_up(Key::F4);
        }

        let output = if cfg!(target_os = "windows") {
            Command::new("powershell")
                .args(&["/C", "(new-object -com shell.application).minimizeall();start myfile.bat -window Maximized"])
                .output()
                .expect("failed to execute process");
        } else {
            Command::new("sh")
                .arg("-c")
                .arg("echo hello")
                .output()
                .expect("failed to execute process");
        };

        for account in 0..number_of_accounts {
            println!("clicked at {:?}", time);
            thread::sleep(Duration::from_secs(2));
            enigo.mouse_move_to(1300, 730);
            enigo.mouse_down(MouseButton::Left);
            enigo.mouse_up(MouseButton::Left);
        }

        thread::sleep(Duration::from_secs(180));
    }
}
use std::process::Command;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use webbrowser;
use rand::Rng;

use enigo::{
    Coordinate,
    Direction::{Click, Press, Release},
    Enigo, Key, Keyboard, Mouse, Settings,
};

pub fn simulate_click_events() {
    let mut enigo = Enigo::new(&Settings::default()).unwrap();
    let now = Instant::now();
    let number_of_accounts:i32 = 7;
    let search_items = ["backpropagation alternatives", "how to get much money",
        "it is possible to get access to browser api?", "be healthy", "are fruits healthy", "what is our future?", "verschwörungstheorie existiert?",
        "recurrent neural network are still good?", "how to get a million dollar?", "statistics forever?", "why are we getting older?",
        "can we live forever?", "brave is the best browser?", "who am I?", "is it good to be faithful to your partner?",
        "why not give people enough money?"
    ];
    let mut rng;
    let time = now.elapsed();
    println!("{:?}", time);

    loop {
        println!("time elapsed {:?}", time);

        if webbrowser::open("https://www.google.de/").is_ok() {
            thread::sleep(Duration::from_secs(2));
            let _ =enigo.key(Key::Return, Click);

            rng = rand::thread_rng();
            let range = 0..search_items.len();
            let random_index = rng.gen_range(range);

            let _= enigo.text(search_items[random_index]);

            let _ = enigo.key(Key::Return, Press);
            thread::sleep(Duration::from_secs(5));
            let _ =enigo.key(Key::Control, Press);
            let _ =enigo.key(Key::F4, Click);
            let _ =enigo.key(Key::Control, Release);
            let _ =enigo.key(Key::F4,Release);
        }

        if cfg!(target_os = "windows") {
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

        for _account in 0..number_of_accounts {
            println!("clicked at {:?}", time);
            thread::sleep(Duration::from_secs(2));
            let _ = enigo.move_mouse(1300, 730, Coordinate::Rel);
        }

        thread::sleep(Duration::from_secs(180));
    }
}
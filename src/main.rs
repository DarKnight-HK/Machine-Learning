use machine_learning::first_example::{cost, train};
use rand::Rng;
fn main() {
    let training_data = [
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
        [5.0, 10.0],
    ];
    let weight: f64 = rand::thread_rng().gen_range(0.0..10.0);
    let bias: f64 = rand::thread_rng().gen_range(0.0..5.0);
    let eps: f64 = 1e-3;
    let learning_rate: f64 = 1e-3;
    let result = cost(weight, &training_data, bias);
    println!("{result}");
    train(500, &training_data, weight, bias, learning_rate, eps);
}

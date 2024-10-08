use rand::Rng;

fn cost(weight: f64, training_data: &[[f64; 2]; 6]) -> f64 {
    let mut result = 0.0;
    for row in training_data.iter() {
        let x = row[0];
        let y = x * weight;
        let error = y - row[1];
        result += error * error;
    }
    result /= training_data.len() as f64;
    result
}

fn main() {
    let training_data = [
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
        [5.0, 10.0],
    ];
    let mut weight: f64 = rand::thread_rng().gen_range(0.0..10.0);
    let eps: f64 = 1e-3;
    let learning_rate: f64 = 1e-3;
    let mut result = cost(weight, &training_data);
    println!("{result}");
    for _ in 0..500 {
        let d_cost = (cost(weight + eps, &training_data) - cost(weight, &training_data)) / eps;
        weight -= d_cost * learning_rate;
        result = cost(weight, &training_data);
    }
    println!("{result}, {weight}");
}

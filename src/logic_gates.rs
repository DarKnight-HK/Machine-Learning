use rand::Rng;

pub fn cost(weight1: f64, weight2: f64, bias: f64, training_data: &[[i32; 3]; 4]) -> f64 {
    let mut result = 0.0;
    for row in training_data.iter() {
        let x1 = row[0];
        let x2 = row[1];
        let y = sigmoid(x1 as f64 * weight1 + x2 as f64 * weight2 + bias);
        let error = y - row[2] as f64;
        result += error * error;
    }
    result /= training_data.len() as f64;
    result
}

fn sigmoid(weight: f64) -> f64 {
    1.0 / (1.0 + (-weight).exp())
}

pub fn train(epochs: i32, training_data: &[[i32; 3]; 4], learning_rate: f64, epsilon: f64) -> f64 {
    let mut weight1: f64 = rand::thread_rng().gen();
    let mut weight2: f64 = rand::thread_rng().gen();
    let mut bias: f64 = rand::thread_rng().gen();
    for _ in 0..epochs {
        let c = cost(weight1, weight2, bias, training_data);
        let dw1 = (cost(weight1 + epsilon, weight2, bias, training_data) - c) / epsilon;
        let dw2 = (cost(weight1, weight2 + epsilon, bias, training_data) - c) / epsilon;
        let db = (cost(weight1, weight2, bias + epsilon, training_data) - c) / epsilon;
        weight1 -= dw1 * learning_rate;
        weight2 -= dw2 * learning_rate;
        bias -= db * learning_rate;
    }
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{i} | {j} = {}",
                sigmoid(i as f64 * weight1 + j as f64 * weight2 + bias)
            )
        }
    }
    cost(weight1, weight2, bias, training_data)
}

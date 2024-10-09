use rand::Rng;
// move the code from the main.rs to the train function
// let training_data = [
//     [0.0, 0.0],
//     [1.0, 2.0],
//     [2.0, 4.0],
//     [3.0, 6.0],
//     [4.0, 8.0],
//     [5.0, 10.0],
// ];
// let eps: f64 = 1e-3;
// let learning_rate: f64 = 1e-3;
// train(1000, &training_data, learning_rate, eps);
pub fn cost(weight: f64, training_data: &[[f64; 2]; 6], bias: f64) -> f64 {
    let mut result = 0.0;
    for row in training_data.iter() {
        let x = row[0];
        let y = x * weight + bias;
        let error = y - row[1];
        result += error * error;
    }
    result /= training_data.len() as f64;
    result
}

pub fn train(epochs: i32, training_data: &[[f64; 2]; 6], learning_rate: f64, epsilon: f64) -> f64 {
    let mut result = 0.0;
    let mut weight: f64 = rand::thread_rng().gen_range(0.0..10.0);
    let mut bias: f64 = rand::thread_rng().gen_range(0.0..5.0);

    for _ in 0..epochs {
        let dw_cost = (cost(weight + epsilon, training_data, bias)
            - cost(weight, training_data, bias))
            / epsilon;
        let db_cost = (cost(weight, training_data, bias + epsilon)
            - cost(weight, training_data, bias))
            / epsilon;

        weight -= dw_cost * learning_rate;
        bias -= db_cost * learning_rate;
        result = cost(weight, training_data, bias);
    }
    println!("Error: {result}");
    println!("Weight: {weight}");
    println!("Bias: {bias}");

    result
}

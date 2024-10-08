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

pub fn train(
    epochs: i32,
    training_data: &[[f64; 2]; 6],
    mut weight: f64,
    mut bias: f64,
    learning_rate: f64,
    epsilon: f64,
) -> f64 {
    let mut result = 0.0;
    for _ in 0..epochs {
        let dw_cost = (cost(weight + epsilon, training_data, bias)
            - cost(weight, training_data, bias))
            / epsilon;
        let db_cost = (cost(weight + epsilon, training_data, bias)
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

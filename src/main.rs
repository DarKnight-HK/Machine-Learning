use machine_learning::logic_gates::train;

fn main() {
    let xor_gate_data = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
    let res = train(2000, &xor_gate_data, 1e-1, 1e-1);
    println!("Error: {res}")
}

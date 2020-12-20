use rand::seq::SliceRandom;
use rand::thread_rng;

use std::io::{self, BufRead};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
};

const FILENAME: &str = "house-votes-84.data";
const ATTRIBUTES_COUNT: usize = 16;
const CROSSVALIDATION_SPLITS: usize = 10;
const CHOICES_COUNT: usize = 3;
const CLASSES_COUNT: usize = 2;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum Choice {
    Yes,
    No,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Class {
    Republican,
    Democrat,
}

#[derive(Debug, Clone)]
struct Row {
    class: Class,
    attributes: Vec<Choice>,
}

fn choice_str_to_enum(c: &str) -> Choice {
    if c == "y" {
        return Choice::Yes;
    }

    if c == "n" {
        return Choice::No;
    }

    Choice::Unknown
}

fn split_for_crossvalidation(mut data: Vec<Row>) -> Vec<Vec<Row>> {
    data.shuffle(&mut thread_rng());
    let chunk_size = data.len() / CROSSVALIDATION_SPLITS;
    let remainder = data.len() % CROSSVALIDATION_SPLITS;

    let mut res: Vec<Vec<Row>> = data.chunks_exact(chunk_size).map(|x| x.to_vec()).collect();
    let mut res_counter = 0;
    let res_len = res.len();

    // Add remainders
    for i in 0..remainder {
        res[res_counter].push(data[res_len + i].clone());
        res_counter += 1;
    }

    res
}

fn read_input() -> Vec<Row> {
    let file = File::open(FILENAME).expect("Couldn't open file");
    let lines = io::BufReader::new(file).lines();
    let mut data: Vec<Row> = vec![];

    for line in lines {
        let line = line.expect("Couldn't read line");
        let split_line: Vec<&str> = line.split(",").collect();

        let class = match split_line[0] {
            "republican" => Class::Republican,
            "democrat" => Class::Democrat,
            _ => panic!("Unknown class"),
        };

        let attributes: Vec<Choice> = split_line[1..=ATTRIBUTES_COUNT]
            .iter()
            .map(|x| choice_str_to_enum(x))
            .collect();

        data.push(Row { class, attributes });
    }

    data
}

#[derive(Debug)]
struct Model {
    attr_probs: HashMap<Class, Vec<HashMap<Choice, f64>>>,
    class_probs: HashMap<Class, f64>,
}

impl Model {
    fn new(data: &Vec<&Row>) -> Self {
        let mut attr_probs: HashMap<Class, Vec<HashMap<Choice, f64>>> = HashMap::new();
        let mut class_probs: HashMap<Class, f64> = HashMap::new();

        attr_probs.insert(Class::Republican, vec![]);
        attr_probs.insert(Class::Democrat, vec![]);

        // Zero out all attribute probabilities
        for (_, attrs) in &mut attr_probs {
            for _ in 0..ATTRIBUTES_COUNT {
                let mut new_hashmap = HashMap::new();
                new_hashmap.insert(Choice::Yes, 1f64 / data.len() as f64);
                new_hashmap.insert(Choice::No, 1f64 / data.len() as f64);
                new_hashmap.insert(Choice::Unknown, 1f64 / data.len() as f64);
                attrs.push(new_hashmap);
            }
        }

        let mut republicans_prob = 0f64;
        let mut democrats_prob = 0f64;

        for row in data {
            if row.class == Class::Republican {
                republicans_prob += 1 as f64 / data.len() as f64;
            } else {
                democrats_prob += 1 as f64 / data.len() as f64;
            }

            for i in 0..ATTRIBUTES_COUNT {
                let choice = row.attributes[i];
                let attribute_prob = &mut attr_probs.get_mut(&row.class).unwrap()[i];
                let choice_prob = attribute_prob.get_mut(&choice).unwrap();
                *choice_prob += 1f64 / data.len() as f64;
            }
        }

        class_probs.insert(Class::Republican, republicans_prob);
        class_probs.insert(Class::Democrat, democrats_prob);

        Model {
            attr_probs,
            class_probs,
        }
    }

    fn predict_class(&self, row: &Row, class: Class) -> f64 {
        let mut res = 0f64;

        res += self.class_probs[&class].log10();

        for i in 0..ATTRIBUTES_COUNT {
            res += self.attr_probs[&class][i][&row.attributes[i]].log10();
        }

        res
    }

    fn predict(&self, row: &Row) -> Class {
        let republican_prob = self.predict_class(row, Class::Republican);
        let democrat_prob = self.predict_class(row, Class::Democrat);

        if republican_prob > democrat_prob {
            return Class::Republican;
        }

        Class::Democrat
    }

    fn get_accuracy(&self, testing_set: &Vec<Row>) -> f64 {
        let mut res = 0f64;

        for row in testing_set {
            let prediction = self.predict(&row);

            if prediction == row.class {
                res += 1f64 / testing_set.len() as f64;
            }
        }

        res
    }
}

fn main() {
    let data = read_input();
    let split_data: Vec<Vec<Row>> = split_for_crossvalidation(data);
    let mut avg_accuracy = None;

    for testing_set_idx in 0..split_data.len() {
        let mut training_set = split_data.clone();
        training_set.remove(testing_set_idx);
        let training_set_merged: Vec<&Row> = training_set.iter().flatten().collect();
        let model = Model::new(&training_set_merged);

        let accuracy = model.get_accuracy(&split_data[testing_set_idx]);
        println!("Accuracy {}: {}", testing_set_idx, accuracy);

        if let Some(average_acc) = &mut avg_accuracy {
            *average_acc += accuracy / CROSSVALIDATION_SPLITS as f64;
        } else {
            avg_accuracy = Some(accuracy / CROSSVALIDATION_SPLITS as f64);
        }
    }

    println!("Average accuracy: {}", avg_accuracy.unwrap());
}

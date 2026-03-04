//! Test Arrow file reading for BEANS-Zero dataset

use std::fs::File;
use std::io::BufReader;
use arrow::ipc::reader::StreamReader;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("beans_zero_data/beans_zero_test/data-00000-of-00291.arrow");

    println!("Opening: {}", path);

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open file: {:?}", e);
            return;
        }
    };
    println!("File opened, size: {} bytes", file.metadata().map(|m| m.len()).unwrap_or(0));

    let buf_reader = BufReader::new(file);
    println!("Creating StreamReader...");

    let reader = match StreamReader::try_new(buf_reader, None) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to create StreamReader: {:?}", e);
            return;
        }
    };

    println!("Reader created successfully!");
    println!("Schema: {:?}", reader.schema());

    let mut batch_count = 0;
    let mut total_rows = 0;

    for batch_result in reader {
        match batch_result {
            Ok(batch) => {
                batch_count += 1;
                total_rows += batch.num_rows();
                println!("Batch {}: {} rows, columns: {:?}",
                    batch_count,
                    batch.num_rows(),
                    batch.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>()
                );

                // Show first row details
                if batch_count == 1 && batch.num_rows() > 0 {
                    for (i, field) in batch.schema().fields().iter().enumerate() {
                        let col = batch.column(i);
                        println!("  Column '{}': {:?}", field.name(), col.data_type());
                    }
                }

                if batch_count >= 3 {
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error reading batch: {:?}", e);
                break;
            }
        }
    }

    println!("\nTotal batches read: {}, total rows: {}", batch_count, total_rows);
}

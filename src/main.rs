use std::collections::{VecDeque, HashMap};
use rand::Rng;
use rand_distr::{Distribution, Exp};
use plotters::prelude::*;

#[derive(Debug)]
enum EventType {
    Arrival { customer_id: u32 },
    Departure { customer_id: u32 },
}

#[derive(Debug)]
struct Event {
    time: f64,
    event_type: EventType,
}

#[derive(Default)]
struct SimulationResult {
    rho: f64,
    simulated_average: f64,
    theoretical_average: f64,
    littles_law_average: f64,
}

struct Simulation {
    lambda: f64,     
    mu: f64,         
    rho: f64,        
    current_time: f64,
    customers_in_system: u32,
    event_queue: VecDeque<Event>,
    total_customers: u32,
    cumulative_time: f64,
    cumulative_customers: f64,
    arrival_times: HashMap<u32, f64>,
    total_response_time: f64,
    completed_customers: u32,
}

impl Simulation {
    fn new(lambda: f64, mu: f64) -> Self {
        Simulation {
            lambda,
            mu,
            rho: lambda / mu,
            current_time: 0.0,
            customers_in_system: 0,
            event_queue: VecDeque::new(),
            total_customers: 0,
            cumulative_time: 0.0,
            cumulative_customers: 0.0,
            arrival_times: HashMap::new(),
            total_response_time: 0.0,
            completed_customers: 0,
        }
    }

    /// Generate next event time using inverse transform method
    fn generate_next_time(&self, rate: f64, previous_time: f64) -> f64 {
        let z: f64 = rand::thread_rng().gen();
        let x = -((1.0 - z).ln()) / rate;
        previous_time + x
    }

    /// Update statistics when state changes
    fn update_stats(&mut self, time_delta: f64) {
        self.cumulative_time += time_delta;
        self.cumulative_customers += self.customers_in_system as f64 * time_delta;
    }

    fn handle_arrival(&mut self, event_time: f64, customer_id: u32) {
        let time_delta = event_time - self.current_time;
        self.update_stats(time_delta);
        
        self.current_time = event_time;
        self.customers_in_system += 1;
        self.total_customers += 1;

        self.arrival_times.insert(customer_id, event_time);

        let next_arrival_time = self.generate_next_time(self.lambda, event_time);
        self.event_queue.push_back(Event {
            time: next_arrival_time,
            event_type: EventType::Arrival { customer_id: customer_id + 1 },
        });

        if self.customers_in_system == 1 {
            let departure_time = self.generate_next_time(self.mu, event_time);
            self.event_queue.push_back(Event {
                time: departure_time,
                event_type: EventType::Departure { customer_id },
            });
        }
    }

    fn handle_departure(&mut self, event_time: f64, customer_id: u32) {
        let time_delta = event_time - self.current_time;
        self.update_stats(time_delta);

        if let Some(arrival_time) = self.arrival_times.remove(&customer_id) {
            let response_time = event_time - arrival_time;
            self.total_response_time += response_time;
            self.completed_customers += 1;
        }

        self.current_time = event_time;
        self.customers_in_system -= 1;

        if self.customers_in_system > 0 {
            let departure_time = self.generate_next_time(self.mu, event_time);
            self.event_queue.push_back(Event {
                time: departure_time,
                event_type: EventType::Departure { customer_id: customer_id + 1 },
            });
        }
    }

    fn get_average_customers(&self) -> f64 {
        if self.cumulative_time > 0.0 {
            self.cumulative_customers / self.cumulative_time
        } else {
            0.0
        }
    }

    fn get_theoretical_average(&self) -> f64 {
        self.rho / (1.0 - self.rho)
    }

    fn get_en_littles_law(&self) -> f64 {
        let avg_response_time = if self.completed_customers > 0 {
            self.total_response_time / self.completed_customers as f64
        } else {
            0.0
        };
        self.lambda * avg_response_time
    }

    fn run(&mut self, total_customers_to_process: u32) {
        // Schedule first arrival
        self.event_queue.push_back(Event {
            time: self.generate_next_time(self.lambda, 0.0),
            event_type: EventType::Arrival { customer_id: 0 },
        });

        while self.completed_customers < total_customers_to_process {
            self.event_queue.make_contiguous();
            let mut events: Vec<_> = self.event_queue.drain(..).collect();
            events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            self.event_queue.extend(events);

            if let Some(event) = self.event_queue.pop_front() {
                match event.event_type {
                    EventType::Arrival { customer_id } => 
                        self.handle_arrival(event.time, customer_id),
                    EventType::Departure { customer_id } => 
                        self.handle_departure(event.time, customer_id),
                }
            }
        }
    }

    fn run_and_get_results(&mut self, total_customers_to_process: u32) -> SimulationResult {
        self.run(total_customers_to_process);
        
        SimulationResult {
            rho: self.rho,
            simulated_average: self.get_average_customers(),
            theoretical_average: self.get_theoretical_average(),
            littles_law_average: self.get_en_littles_law(),
        }
    }

    fn print_results(&self) {
        println!("\nSimulation Results:");
        println!("Load (ρ) = {:.3}", self.rho);
        println!("Total customers processed: {}", self.completed_customers);
        println!("Average response time E[D]: {:.3}", 
                self.total_response_time / self.completed_customers as f64);
        println!("E[N] (direct measurement): {:.3}", self.get_average_customers());
        println!("E[N] (Little's Law): {:.3}", self.get_en_littles_law());
        println!("Theoretical E[N]: {:.3}", self.get_theoretical_average());
        println!("Total simulation time: {:.3}", self.cumulative_time);
    }
}

fn create_comparison_plot(results: &[SimulationResult]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("mm1_queue_results.png", (800, 600))
        .into_drawing_area();
    
    root.fill(&WHITE)?;

    let max_y = results.iter()
        .map(|r| r.simulated_average.max(r.theoretical_average).max(r.littles_law_average))
        .fold(f64::NEG_INFINITY, f64::max) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("M/M/1 Queue Simulation Results", ("sans-serif", 30).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..1f64, 0f64..max_y)?;

    chart
        .configure_mesh()
        .x_desc("Load (ρ)")
        .y_desc("Average Number of Customers")
        .draw()?;

    // Plot theoretical values
    chart.draw_series(LineSeries::new(
        results.iter().map(|r| (r.rho, r.theoretical_average)),
        RED.filled(),
    ))?.label("Theoretical")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    // Plot simulated values
    chart.draw_series(results.iter().map(|r| {
        Circle::new((r.rho, r.simulated_average), 5, BLUE.filled())
    }))?.label("Simulated")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, BLUE.filled()));

    // Plot Little's Law values
    chart.draw_series(results.iter().map(|r| {
        Circle::new((r.rho, r.littles_law_average), 5, GREEN.filled())
    }))?.label("Little's Law")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, GREEN.filled()));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mu = 1.0;
    let loads = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let mut results = Vec::new();

    for &rho in &loads {
        println!("\nRunning simulation for ρ = {}", rho);
        let lambda = rho * mu;
        let mut simulation = Simulation::new(lambda, mu);
        let result = simulation.run_and_get_results(10000);
        simulation.print_results();
        results.push(result);
    }

    create_comparison_plot(&results)?;
    println!("\nPlot has been saved as 'mm1_queue_results.png'");

    Ok(())
}
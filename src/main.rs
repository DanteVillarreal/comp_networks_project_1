use std::collections::VecDeque;
use rand::Rng;
use rand_distr::{Distribution, Exp};
use plotters::prelude::*;

/// Structure to hold simulation results
#[derive(Default)]
struct SimulationResult {
    rho: f64,
    simulated_average: f64,
    theoretical_average: f64,
}
#[derive(Debug)]
enum EventType {
    Arrival { customer_id: u32 },    // Added customer_id to track arrivals
    Departure { customer_id: u32 },   // Added customer_id to track departures
}

#[derive(Debug)]
struct Event {
    time: f64,
    event_type: EventType,
}

struct Simulation {
    // Previous fields remain...
    lambda: f64,     
    mu: f64,         
    rho: f64,        
    current_time: f64,
    customers_in_system: u32,
    event_queue: VecDeque<Event>,
    total_customers: u32,
    cumulative_time: f64,
    cumulative_customers: f64,
    
    // New fields for tracking customer response times
    arrival_times: std::collections::HashMap<u32, f64>,
    total_response_time: f64,
    completed_customers: u32,
}

impl Simulation {
    /// Create a new simulation with given arrival and service rates
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
            arrival_times: std::collections::HashMap::new(),
            total_response_time: 0.0,
            completed_customers: 0,
        }
    }

    // /// Schedule an initial arrival event
    // fn schedule_initial_arrival(&mut self) {
    //     let arrival_time = self.generate_interarrival_time();
    //     self.event_queue.push_back(Event {
    //         time: arrival_time,
    //         event_type: EventType::Arrival { customer_id: 0 },  // First customer has ID 0
    //     });
    // }

    // /// Generate time until next arrival (exponential with rate lambda)
    // fn generate_interarrival_time(&self) -> f64 {
    //     let exp = Exp::new(self.lambda).unwrap();
    //     exp.sample(&mut rand::thread_rng())
    // }

    // /// Generate service time (exponential with rate mu)
    // fn generate_service_time(&self) -> f64 {
    //     let exp = Exp::new(self.mu).unwrap();
    //     exp.sample(&mut rand::thread_rng())
    // }

    
    /// Generate next event time using inverse transform method
    fn generate_next_time(&self, rate: f64, previous_time: f64) -> f64 {
        // 1. Generate uniform random number in [0,1]
        let z: f64 = rand::thread_rng().gen();
        
        // 2. Apply inverse transform: F^(-1)(z) = -ln(1-z)/rate for exponential distribution
        let x = -((1.0 - z).ln()) / rate;
        
        // 3. ti = x + ti-1
        previous_time + x
    }

    /// Update statistics when state changes
    fn update_stats(&mut self, time_delta: f64) {
        self.cumulative_time += time_delta;
        self.cumulative_customers += self.customers_in_system as f64 * time_delta;
    }

    /// Get average number of customers in system
    fn get_average_customers(&self) -> f64 {
        if self.cumulative_time > 0.0 {
            self.cumulative_customers / self.cumulative_time
        } else {
            0.0
        }
    }

    /// Get theoretical average number of customers
    fn get_theoretical_average(&self) -> f64 {
        self.rho / (1.0 - self.rho)
    }
    
    /// Handle an arrival event
    fn handle_arrival(&mut self, event_time: f64, customer_id: u32) {
        // Update statistics
        let time_delta = event_time - self.current_time;
        self.update_stats(time_delta);
        
        // Update current time and increment customers
        self.current_time = event_time;
        self.customers_in_system += 1;
        self.total_customers += 1;

        // Store arrival time for response time calculation
        self.arrival_times.insert(customer_id, event_time);

        // Schedule next arrival
        let next_arrival_time = self.generate_next_time(self.lambda, event_time);
        self.event_queue.push_back(Event {
            time: next_arrival_time,
            event_type: EventType::Arrival { customer_id: customer_id + 1 },
        });

        // If this is the only customer, schedule their departure
        if self.customers_in_system == 1 {
            let departure_time = self.generate_next_time(self.mu, event_time);
            self.event_queue.push_back(Event {
                time: departure_time,
                event_type: EventType::Departure { customer_id },
            });
        }
    }
    /// Handle a departure event
    fn handle_departure(&mut self, event_time: f64, customer_id: u32) {
        // Update statistics
        let time_delta = event_time - self.current_time;
        self.update_stats(time_delta);

        // Calculate response time for this customer
        if let Some(arrival_time) = self.arrival_times.remove(&customer_id) {
            let response_time = event_time - arrival_time;
            self.total_response_time += response_time;
            self.completed_customers += 1;
        }

        // Update current time and decrement customers
        self.current_time = event_time;
        self.customers_in_system -= 1;

        // If there are more customers, schedule next departure
        if self.customers_in_system > 0 {
            let departure_time = self.generate_next_time(self.mu, event_time);
            self.event_queue.push_back(Event {
                time: departure_time,
                event_type: EventType::Departure { 
                    customer_id: customer_id + 1 
                },
            });
        }
    }
    /// Calculate E[N] using Little's Law
    fn get_en_littles_law(&self) -> f64 {
        let avg_response_time = if self.completed_customers > 0 {
            self.total_response_time / self.completed_customers as f64
        } else {
            0.0
        };
        self.lambda * avg_response_time  // E[N] = λ * E[D]
    }
    fn run(&mut self, total_customers_to_process: u32) {
        // Schedule first arrival
        self.event_queue.push_back(Event {
            time: self.generate_next_time(self.lambda, 0.0),
            event_type: EventType::Arrival { customer_id: 0 },
        });

        // Process events until we've served enough customers
        while self.completed_customers < total_customers_to_process {
            // Sort events by time
            self.event_queue.make_contiguous();
            let mut events: Vec<_> = self.event_queue.drain(..).collect();
            events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            self.event_queue.extend(events);

            // Process next event
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

    fn print_results(&self) {
        println!("\nSimulation Results:");
        println!("Load (ρ) = {:.3}", self.rho);
        println!("Total customers processed: {}", self.completed_customers);
        println!("Average response time E[D]: {:.3}", 
                self.total_response_time / self.completed_customers as f64);
        println!("E[N] (direct measurement): {:.3}", self.get_average_customers());
        println!("E[N] (Little's Law): {:.3}", self.get_en_littles_law());
        println!("Theoretical E[N]: {:.3}", self.get_theoretical_average());
    }

    /// Run simulation and return results
    fn run_and_get_results(&mut self, total_customers_to_process: u32) -> SimulationResult {
        self.run(total_customers_to_process);
        
        SimulationResult {
            rho: self.rho,
            simulated_average: self.get_average_customers(),
            theoretical_average: self.get_theoretical_average(),
        }
    }
}
/// Function to create plot comparing theoretical and simulated results
fn create_comparison_plot(results: &[SimulationResult]) -> Result<(), Box<dyn std::error::Error>> {
    // Create plot area
    let root = BitMapBackend::new("mm1_queue_results.png", (800, 600))
        .into_drawing_area();
    
    root.fill(&WHITE)?;

    // Find max y value for plot scaling
    let max_y = results.iter()
        .map(|r| r.simulated_average.max(r.theoretical_average))
        .fold(f64::NEG_INFINITY, f64::max) * 1.1;

    // Create chart
    let mut chart = ChartBuilder::on(&root)
        .caption("M/M/1 Queue Simulation Results", ("sans-serif", 30).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..1f64, 0f64..max_y)?;

    // Configure chart
    chart
        .configure_mesh()
        .x_desc("Load (ρ)")
        .y_desc("Average Number of Customers")
        .draw()?;

    // Plot theoretical values (line)
    chart.draw_series(LineSeries::new(
        results.iter().map(|r| (r.rho, r.theoretical_average)),
        RED.filled(),
    ))?.label("Theoretical")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    // Plot simulated values (points)
    chart.draw_series(results.iter().map(|r| {
        Circle::new(
            (r.rho, r.simulated_average),
            5,
            BLUE.filled(),
        )
    }))?.label("Simulated")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, BLUE.filled()));

    // Draw legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mu = 1.0;  // Fixed service rate
    let loads = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let mut results = Vec::new();

    // Run simulations for different loads
    for &rho in &loads {
        println!("\nRunning simulation for ρ = {}", rho);
        let lambda = rho * mu;
        let mut simulation = Simulation::new(lambda, mu);
        let result = simulation.run_and_get_results(10000);
        simulation.print_results();
        results.push(result);
    }

    // Create plot
    create_comparison_plot(&results)?;
    println!("\nPlot has been saved as 'mm1_queue_results.png'");

    Ok(())
}
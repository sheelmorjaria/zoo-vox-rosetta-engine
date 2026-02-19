//! Meerkat Duration/Rate Context Encoding Analysis
//!
//! Tests whether context is encoded by DELIVERY PARAMETERS rather than
//! phrase type or sequence. This follows the bat analysis which showed
//! that duration explains 70.4% of repetition variance.
//!
//! Hypothesis: Context is encoded by HOW phrases are delivered
//! (duration, rate) not WHAT phrases are used.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Write};

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Clone)]
pub struct DurationStats {
    pub mean_ms: f64,
    pub median_ms: f64,
    pub std_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub n_samples: usize,
}

#[derive(Debug, Clone)]
pub struct RateStats {
    pub phrases_per_second: f64,
    pub phrases_per_call: f64,
    pub mean_call_duration_ms: f64,
    pub n_calls: usize,
    pub total_phrases: usize,
}

#[derive(Debug, Clone)]
pub struct AnovaResult {
    pub f_statistic: f64,
    pub p_value: f64,
    pub significant: bool,
    pub effect_size: f64, // eta-squared
    pub df_between: usize,
    pub df_within: usize,
}

#[derive(Debug, Clone)]
pub struct CorrelationResult {
    pub r: f64,
    pub r_squared: f64,
    pub p_value: f64,
    pub n_samples: usize,
    pub interpretation: String,
}

#[derive(Debug)]
pub struct DeliveryEncodingAnalysis {
    /// Duration statistics by context
    pub duration_by_context: HashMap<String, DurationStats>,

    /// Phrase rate by context
    pub rate_by_context: HashMap<String, RateStats>,

    /// ANOVA: Duration differs by context?
    pub duration_anova: AnovaResult,

    /// ANOVA: Phrase count differs by context?
    pub phrase_count_anova: AnovaResult,

    /// Correlation: Duration vs Phrase Count (like bat ANCOVA)
    pub duration_phrase_correlation: CorrelationResult,

    /// Total samples
    pub total_calls: usize,
    pub total_contexts: usize,
}

// =============================================================================
// Label mappings
// =============================================================================

const LABEL_MEANINGS: &[(&str, &str)] = &[
    ("cc", "Close Call"),
    ("sn", "Sentinel"),
    ("soc", "Social"),
    ("oth", "Other"),
    ("agg", "Aggression"),
    ("synch", "Synchronized"),
    ("al", "Alarm"),
    ("eating", "Eating"),
    ("mo", "Movement"),
    ("beep", "Calibration"),
    ("ld", "Lead"),
];

fn get_label_meaning(label: &str) -> &str {
    LABEL_MEANINGS
        .iter()
        .find(|(l, _)| *l == label)
        .map(|(_, m)| *m)
        .unwrap_or(label)
}

// =============================================================================
// Analysis Functions
// =============================================================================

fn load_phrase_data(path: &str) -> Vec<(String, f64, usize)> {
    // Returns (filename, duration_ms, phrase_count)
    let file = File::open(path).expect("Failed to open phrase data");
    let reader = BufReader::new(file);

    let json: serde_json::Value = serde_json::from_reader(reader).expect("Failed to parse JSON");

    let mut results = Vec::new();

    if let Some(arr) = json.as_array() {
        for item in arr {
            if let Some(file_name) = item.get("file_name").and_then(|v| v.as_str()) {
                let fname = file_name.replace(".wav", "");

                // Get duration from analysis - field is "total_duration_ms"
                let duration_ms = item
                    .get("total_duration_ms")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                // Count phrases
                let phrase_count = item
                    .get("phrases")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.len())
                    .unwrap_or(0);

                if duration_ms > 0.0 && phrase_count > 0 {
                    results.push((fname, duration_ms, phrase_count));
                }
            }
        }
    }

    results
}

fn load_labels_via_python(labels_dir: &str) -> HashMap<String, String> {
    use std::process::{Command, Stdio};

    println!("   🔄 Loading labels via Python subprocess...");

    let python_script = r#"
import h5py
import os
import json
import sys

labels_dir = sys.argv[1]
output = {}

for lbl_file in os.listdir(labels_dir):
    if not lbl_file.endswith('.h5'):
        continue

    fname = lbl_file.replace('.h5', '')
    path = os.path.join(labels_dir, lbl_file)

    try:
        with h5py.File(path, 'r') as f:
            lbls = f['lbl'][:]
            if len(lbls) > 0:
                lbls_str = [l.decode() if isinstance(l, bytes) else l for l in lbls]
                from collections import Counter
                lbl_counter = Counter(lbls_str)
                primary = lbl_counter.most_common(1)[0][0]

                output[fname] = primary
    except:
        pass

print(json.dumps(output))
"#;

    let temp_script = "/tmp/load_meerkat_labels_delivery.py";
    let mut file = File::create(temp_script).expect("Failed to create temp script");
    file.write_all(python_script.as_bytes())
        .expect("Failed to write script");

    let output = Command::new("python3")
        .arg(temp_script)
        .arg(labels_dir)
        .stdout(Stdio::piped())
        .output()
        .expect("Failed to run Python script");

    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut labels = HashMap::new();
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
        if let Some(obj) = json.as_object() {
            for (k, v) in obj {
                if let Some(label) = v.as_str() {
                    labels.insert(k.clone(), label.to_string());
                }
            }
        }
    }

    labels
}

fn compute_anova(groups: &HashMap<String, Vec<f64>>) -> AnovaResult {
    // Grand mean
    let all_values: Vec<f64> = groups.values().flat_map(|v| v.iter().cloned()).collect();
    if all_values.is_empty() {
        return AnovaResult {
            f_statistic: 0.0,
            p_value: 1.0,
            significant: false,
            effect_size: 0.0,
            df_between: 0,
            df_within: 0,
        };
    }

    let grand_mean = all_values.iter().sum::<f64>() / all_values.len() as f64;
    let n_total = all_values.len();
    let k = groups.len();

    // Between-group sum of squares
    let ss_between: f64 = groups
        .values()
        .map(|group| {
            let group_mean = group.iter().sum::<f64>() / group.len() as f64;
            group.len() as f64 * (group_mean - grand_mean).powi(2)
        })
        .sum();

    // Within-group sum of squares
    let ss_within: f64 = groups
        .values()
        .map(|group| {
            let group_mean = group.iter().sum::<f64>() / group.len() as f64;
            group.iter().map(|x| (x - group_mean).powi(2)).sum::<f64>()
        })
        .sum();

    let df_between = k - 1;
    let df_within = n_total - k;

    if df_within == 0 {
        return AnovaResult {
            f_statistic: 0.0,
            p_value: 1.0,
            significant: false,
            effect_size: 0.0,
            df_between,
            df_within,
        };
    }

    let ms_between = ss_between / df_between as f64;
    let ms_within = ss_within / df_within as f64;

    let f = if ms_within > 0.0 {
        ms_between / ms_within
    } else {
        0.0
    };

    // Approximate p-value using F-distribution approximation
    let p = approximate_f_pvalue(f, df_between as f64, df_within as f64);

    AnovaResult {
        f_statistic: f,
        p_value: p,
        significant: p < 0.05,
        effect_size: ss_between / (ss_between + ss_within), // eta-squared
        df_between,
        df_within,
    }
}

fn approximate_f_pvalue(f: f64, df1: f64, df2: f64) -> f64 {
    // Approximate p-value for F-distribution
    // Using a simple approximation based on critical values
    if f < 1.0 {
        return 0.5;
    }

    // Critical F values at alpha=0.05 for various df combinations
    let critical_05 = match (df1 as usize, df2 as usize) {
        (1, n) if n > 100 => 3.94,
        (1, n) if n > 30 => 4.17,
        (1, _) => 4.30,
        (2, n) if n > 100 => 3.09,
        (2, n) if n > 30 => 3.32,
        (2, _) => 3.49,
        (3, n) if n > 100 => 2.70,
        (3, n) if n > 30 => 2.92,
        (3, _) => 3.10,
        (10, n) if n > 100 => 1.93,
        (10, _) => 2.16,
        _ => 2.0, // Default approximation
    };

    let critical_01 = critical_05 * 1.3;
    let critical_001 = critical_05 * 1.7;

    if f > critical_001 {
        0.0001
    } else if f > critical_01 {
        0.001
    } else if f > critical_05 {
        0.01
    } else {
        0.1
    }
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> CorrelationResult {
    let n = x.len().min(y.len());
    if n < 3 {
        return CorrelationResult {
            r: 0.0,
            r_squared: 0.0,
            p_value: 1.0,
            n_samples: n,
            interpretation: "Insufficient data".to_string(),
        };
    }

    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let r = if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x * var_y).sqrt()
    } else {
        0.0
    };

    let r_squared = r * r;

    // t-test for correlation
    let t = if r.abs() < 1.0 {
        r * ((n - 2) as f64 / (1.0 - r_squared)).sqrt()
    } else {
        100.0 // Perfect correlation
    };

    // Approximate p-value
    let p = if t.abs() > 10.0 {
        0.0001
    } else if t.abs() > 5.0 {
        0.001
    } else if t.abs() > 3.0 {
        0.01
    } else if t.abs() > 2.0 {
        0.05
    } else {
        0.1
    };

    let interpretation = if r > 0.7 {
        format!(
            "Strong positive (R²={:.1}% variance explained)",
            r_squared * 100.0
        )
    } else if r > 0.4 {
        format!(
            "Moderate positive (R²={:.1}% variance explained)",
            r_squared * 100.0
        )
    } else if r > 0.2 {
        format!(
            "Weak positive (R²={:.1}% variance explained)",
            r_squared * 100.0
        )
    } else if r > -0.2 {
        format!(
            "Negligible (R²={:.1}% variance explained)",
            r_squared * 100.0
        )
    } else if r > -0.4 {
        format!(
            "Weak negative (R²={:.1}% variance explained)",
            r_squared * 100.0
        )
    } else if r > -0.7 {
        format!(
            "Moderate negative (R²={:.1}% variance explained)",
            r_squared * 100.0
        )
    } else {
        format!(
            "Strong negative (R²={:.1}% variance explained)",
            r_squared * 100.0
        )
    };

    CorrelationResult {
        r,
        r_squared,
        p_value: p,
        n_samples: n,
        interpretation,
    }
}

fn analyze_delivery_encoding(
    call_data: &[(String, f64, usize)],
    labels: &HashMap<String, String>,
) -> DeliveryEncodingAnalysis {
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Organizing Data by Context                              │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // Group by context
    let mut duration_by_context: HashMap<String, Vec<f64>> = HashMap::new();
    let mut phrases_by_context: HashMap<String, Vec<usize>> = HashMap::new();

    for (fname, duration, phrase_count) in call_data {
        if let Some(context) = labels.get(fname) {
            duration_by_context
                .entry(context.clone())
                .or_default()
                .push(*duration);
            phrases_by_context
                .entry(context.clone())
                .or_default()
                .push(*phrase_count);
        }
    }

    println!("   Matched {} calls with context labels", call_data.len());
    println!("   Found {} unique contexts", duration_by_context.len());

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Computing Duration Statistics by Context                │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // Duration stats by context
    let mut duration_stats: HashMap<String, DurationStats> = HashMap::new();

    println!("\n   📊 Duration by Context:");
    println!("   ┌────────────────────┬──────────┬──────────┬──────────┬────────────┐");
    println!("   │ Context            │ Mean (ms)│ Std (ms) │ Median   │ N calls    │");
    println!("   ├────────────────────┼──────────┼──────────┼──────────┼────────────┤");

    let mut contexts_sorted: Vec<_> = duration_by_context.iter().collect();
    contexts_sorted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (ctx, durations) in &contexts_sorted {
        let n = durations.len();
        let mean = durations.iter().sum::<f64>() / n as f64;
        let median = {
            let mut sorted: Vec<f64> = durations.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[n / 2]
        };
        let std = (durations.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let min = durations.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        duration_stats.insert(
            (*ctx).clone(),
            DurationStats {
                mean_ms: mean,
                median_ms: median,
                std_ms: std,
                min_ms: min,
                max_ms: max,
                n_samples: n,
            },
        );

        let meaning = get_label_meaning(ctx);
        println!(
            "   │ {:<18} │ {:>8.1} │ {:>8.1} │ {:>8.1} │ {:>10} │",
            format!("{} ({})", ctx, meaning),
            mean,
            std,
            median,
            n
        );
    }
    println!("   └────────────────────┴──────────┴──────────┴──────────┴────────────┘");

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Computing Phrase Rate by Context                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // Rate stats by context
    let mut rate_stats: HashMap<String, RateStats> = HashMap::new();

    println!("\n   📊 Phrase Rate by Context:");
    println!("   ┌────────────────────┬────────────┬────────────┬────────────┐");
    println!("   │ Context            │ Phr/Call   │ Phr/Second │ Total Phr  │");
    println!("   ├────────────────────┼────────────┼────────────┼────────────┤");

    for (ctx, durations) in &contexts_sorted {
        let phrase_counts = &phrases_by_context[*ctx];
        let total_phrases: usize = phrase_counts.iter().sum();
        let total_duration_sec: f64 = durations.iter().sum::<f64>() / 1000.0;
        let n_calls = durations.len();

        let phrases_per_call = total_phrases as f64 / n_calls as f64;
        let phrases_per_second = total_phrases as f64 / total_duration_sec;
        let mean_duration = durations.iter().sum::<f64>() / n_calls as f64;

        rate_stats.insert(
            (*ctx).clone(),
            RateStats {
                phrases_per_second,
                phrases_per_call,
                mean_call_duration_ms: mean_duration,
                n_calls,
                total_phrases,
            },
        );

        let meaning = get_label_meaning(ctx);
        println!(
            "   │ {:<18} │ {:>10.2} │ {:>10.2} │ {:>10} │",
            format!("{} ({})", ctx, meaning),
            phrases_per_call,
            phrases_per_second,
            total_phrases
        );
    }
    println!("   └────────────────────┴────────────┴────────────┴────────────┘");

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Statistical Tests                                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // ANOVA for duration
    let duration_anova = compute_anova(&duration_by_context);
    println!("\n   📈 ANOVA: Duration by Context");
    println!(
        "      ├─ F({}, {}) = {:.2}",
        duration_anova.df_between, duration_anova.df_within, duration_anova.f_statistic
    );
    println!("      ├─ p-value = {:.4}", duration_anova.p_value);
    println!(
        "      ├─ Effect size (η²) = {:.4}",
        duration_anova.effect_size
    );
    println!(
        "      └─ Significant: {}",
        if duration_anova.significant {
            "YES ✓"
        } else {
            "NO ✗"
        }
    );

    // Convert phrase counts to f64 for ANOVA
    let phrases_f64: HashMap<String, Vec<f64>> = phrases_by_context
        .iter()
        .map(|(k, v)| (k.clone(), v.iter().map(|&x| x as f64).collect()))
        .collect();
    let phrase_count_anova = compute_anova(&phrases_f64);
    println!("\n   📈 ANOVA: Phrase Count by Context");
    println!(
        "      ├─ F({}, {}) = {:.2}",
        phrase_count_anova.df_between, phrase_count_anova.df_within, phrase_count_anova.f_statistic
    );
    println!("      ├─ p-value = {:.4}", phrase_count_anova.p_value);
    println!(
        "      ├─ Effect size (η²) = {:.4}",
        phrase_count_anova.effect_size
    );
    println!(
        "      └─ Significant: {}",
        if phrase_count_anova.significant {
            "YES ✓"
        } else {
            "NO ✗"
        }
    );

    // Duration-Phrase correlation (like bat ANCOVA)
    let durations: Vec<f64> = call_data.iter().map(|(_, d, _)| *d).collect();
    let phrases: Vec<f64> = call_data.iter().map(|(_, _, p)| *p as f64).collect();
    let duration_phrase_correlation = pearson_correlation(&durations, &phrases);

    println!("\n   📈 Correlation: Duration vs Phrase Count (ANCOVA-style)");
    println!("      ├─ r = {:.4}", duration_phrase_correlation.r);
    println!(
        "      ├─ R² = {:.4} ({:.1}% variance explained)",
        duration_phrase_correlation.r_squared,
        duration_phrase_correlation.r_squared * 100.0
    );
    println!(
        "      ├─ p-value = {:.4}",
        duration_phrase_correlation.p_value
    );
    println!("      ├─ n = {}", duration_phrase_correlation.n_samples);
    println!("      └─ {}", duration_phrase_correlation.interpretation);

    DeliveryEncodingAnalysis {
        duration_by_context: duration_stats,
        rate_by_context: rate_stats,
        duration_anova,
        phrase_count_anova,
        duration_phrase_correlation,
        total_calls: call_data.len(),
        total_contexts: duration_by_context.len(),
    }
}

// =============================================================================
// Output and Reporting
// =============================================================================

impl DeliveryEncodingAnalysis {
    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║       DELIVERY ENCODING ANALYSIS RESULTS                       ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        println!("\n📊 SUMMARY:");
        println!("   ├─ Total calls analyzed: {}", self.total_calls);
        println!("   └─ Behavioral contexts: {}", self.total_contexts);

        // Hypothesis test
        self.print_hypothesis_conclusion();
    }

    fn print_hypothesis_conclusion(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║           HYPOTHESIS TEST: DELIVERY ENCODING                   ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        println!("\n   📈 Key Findings:");

        // Duration ANOVA
        if self.duration_anova.significant {
            println!("\n   ✅ DURATION DIFFERS BY CONTEXT (p < 0.05)");
            println!(
                "      Effect size: {:.1}% of variance explained",
                self.duration_anova.effect_size * 100.0
            );
        } else {
            println!(
                "\n   ❌ Duration does NOT differ by context (p = {:.4})",
                self.duration_anova.p_value
            );
        }

        // Phrase count ANOVA
        if self.phrase_count_anova.significant {
            println!("\n   ✅ PHRASE COUNT DIFFERS BY CONTEXT (p < 0.05)");
            println!(
                "      Effect size: {:.1}% of variance explained",
                self.phrase_count_anova.effect_size * 100.0
            );
        } else {
            println!(
                "\n   ❌ Phrase count does NOT differ by context (p = {:.4})",
                self.phrase_count_anova.p_value
            );
        }

        // Duration-repetition correlation
        println!(
            "\n   📊 DURATION-REPETITION CORRELATION: r = {:.3}",
            self.duration_phrase_correlation.r
        );
        println!(
            "      R² = {:.1}% of phrase count variance explained by duration",
            self.duration_phrase_correlation.r_squared * 100.0
        );

        // Overall conclusion
        println!("\n   🔬 OVERALL INTERPRETATION:");

        if self.duration_anova.significant || self.phrase_count_anova.significant {
            println!();
            println!("   ✅ EVIDENCE FOR DELIVERY-BASED ENCODING");
            println!();
            println!("      Context appears to be encoded by HOW vocalizations are");
            println!("      delivered (duration, complexity) rather than WHAT phrases");
            println!("      are used or their sequence.");
            println!();
            if self.duration_phrase_correlation.r > 0.5 {
                println!(
                    "      The strong duration-repetition correlation (r={:.2}) suggests",
                    self.duration_phrase_correlation.r
                );
                println!("      longer calls naturally contain more phrases, and different");
                println!("      contexts may require different call lengths.");
            }
        } else {
            println!();
            println!("   ❌ WEAK EVIDENCE FOR DELIVERY-BASED ENCODING");
            println!();
            println!("      Neither duration nor phrase count differ significantly");
            println!("      across contexts. Context encoding may rely on:");
            println!("      - Acoustic features (spectral, formants)");
            println!("      - Amplitude/prosody");
            println!("      - Multi-modal signals (visual, olfactory)");
            println!("      - Higher sample rate data (>8kHz)");
        }

        // Comparison with bats
        println!("\n   📊 COMPARISON WITH EGYPTIAN FRUIT BATS:");
        println!("      Bats: duration explains 70.4% of repetition variance");
        println!(
            "      Meerkats: duration explains {:.1}% of repetition variance",
            self.duration_phrase_correlation.r_squared * 100.0
        );
    }

    pub fn save_results(&self, output_path: &str) -> std::io::Result<()> {
        use serde_json::json;

        let results = json!({
            "summary": {
                "total_calls": self.total_calls,
                "total_contexts": self.total_contexts,
            },
            "duration_anova": {
                "f_statistic": self.duration_anova.f_statistic,
                "p_value": self.duration_anova.p_value,
                "significant": self.duration_anova.significant,
                "effect_size": self.duration_anova.effect_size,
            },
            "phrase_count_anova": {
                "f_statistic": self.phrase_count_anova.f_statistic,
                "p_value": self.phrase_count_anova.p_value,
                "significant": self.phrase_count_anova.significant,
                "effect_size": self.phrase_count_anova.effect_size,
            },
            "duration_phrase_correlation": {
                "r": self.duration_phrase_correlation.r,
                "r_squared": self.duration_phrase_correlation.r_squared,
                "p_value": self.duration_phrase_correlation.p_value,
                "interpretation": self.duration_phrase_correlation.interpretation,
            },
            "duration_by_context": self.duration_by_context.iter()
                .map(|(k, v)| (k.clone(), json!({
                    "mean_ms": v.mean_ms,
                    "std_ms": v.std_ms,
                    "median_ms": v.median_ms,
                    "n_samples": v.n_samples,
                })))
                .collect::<HashMap<_, _>>(),
            "rate_by_context": self.rate_by_context.iter()
                .map(|(k, v)| (k.clone(), json!({
                    "phrases_per_call": v.phrases_per_call,
                    "phrases_per_second": v.phrases_per_second,
                    "n_calls": v.n_calls,
                })))
                .collect::<HashMap<_, _>>(),
        });

        let mut file = File::create(output_path)?;
        file.write_all(serde_json::to_string_pretty(&results)?.as_bytes())?;

        println!("\n   ✓ Saved to: {}", output_path);
        Ok(())
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   Meerkat Duration/Rate Context Encoding Analysis              ║");
    println!("╠═════════════════════════════════════════════════════════════════╣");
    println!("║                                                                 ║");
    println!("║  HYPOTHESIS: Context is encoded by DELIVERY parameters         ║");
    println!("║                                                                 ║");
    println!("║  Since phrase TYPE and SEQUENCE don't vary by context,         ║");
    println!("║  this tests if HOW phrases are delivered matters.              ║");
    println!("║                                                                 ║");
    println!("║  Like bat ANCOVA: duration explains 70.4% of repetition        ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");

    let phrase_path = "/mnt/c/Users/sheel/Desktop/data/MeerKAT_10s_2024-06-12/within_call_results/meerkat_within_call_analyses.json";
    let labels_dir = "/mnt/c/Users/sheel/Desktop/data/MeerKAT_10s_2024-06-12/lbl/08000Hz";
    let output_dir =
        "/mnt/c/Users/sheel/Desktop/data/MeerKAT_10s_2024-06-12/delivery_encoding_results";

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Data                                            │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("   📂 Loading within-call phrase data...");
    let call_data = load_phrase_data(phrase_path);
    println!(
        "      └─ Loaded {} calls with duration/phrase data",
        call_data.len()
    );

    let labels = load_labels_via_python(labels_dir);
    println!("      └─ Loaded {} file labels", labels.len());

    // Run analysis
    let analysis = analyze_delivery_encoding(&call_data, &labels);

    // Print summary
    analysis.print_summary();

    // Save results
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Saving Results                                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    std::fs::create_dir_all(output_dir).ok();
    let output_path = format!("{}/delivery_encoding_analysis.json", output_dir);
    analysis.save_results(&output_path).ok();

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║ ANALYSIS COMPLETE                                               ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
}

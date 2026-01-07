// Multi-Node Coordination
//
// Enables arrays of devices to coordinate:
// - Time synchronization via PTP grandmaster election
// - TDMA for acoustic interference avoidance
// - Data fusion from multiple nodes

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::ptp::PtpTimestamp;

/// Node identifier
pub type NodeId = String;

/// Cluster identifier
pub type ClusterId = String;

/// PTP Clock Class (IEEE 1588-2008)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ClockClass {
    /// Clock class 6 - Primary reference time source (GPS, atomic clock)
    ClockClass6 = 6,
    /// Clock class 7 - Network time protocol
    ClockClass7 = 7,
    /// Clock class 13 - Free-running clock
    ClockClass13 = 13,
    /// Clock class 248 - Default for ordinary clocks
    ClockClass248 = 248,
}

/// PTP Clock Accuracy (IEEE 1588-2008)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ClockAccuracy {
    /// Nanosecond accuracy
    Nanosecond25 = 0x20,
    /// Microsecond accuracy
    Microsecond1 = 0x21,
    /// Millisecond accuracy
    Millisecond1 = 0x30,
    /// Second accuracy
    Second1 = 0x40,
}

/// PTP grandmaster election result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElectionResult {
    ElectedGrandmaster,
    NotElected,
    AlreadyGrandmaster,
}

/// TDMA slot assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdmaSlot {
    pub node_id: NodeId,
    pub slot_index: u32,
    pub start_time_us: u64,  // Microseconds from epoch
    pub duration_us: u64,
    pub guard_time_us: u64,
}

/// TDMA schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdmaSchedule {
    pub slots: Vec<TdmaSlot>,
    pub frame_duration_us: u64,
    pub epoch: PtpTimestamp,
}

/// Node information for cluster coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: NodeId,
    pub cluster_id: ClusterId,
    pub clock_class: ClockClass,
    pub clock_accuracy: ClockAccuracy,
    pub priority: u8,  // 1-255, lower is higher priority
    pub last_seen: PtpTimestamp,
    pub capabilities: NodeCapabilities,
}

/// Node capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub can_be_grandmaster: bool,
    pub has_gps: bool,
    pub has_atomic_clock: bool,
    pub supports_tdma: bool,
    pub max_sample_rate: u32,
}

/// Fused data from multiple nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedDetectionData {
    pub primary_node_id: NodeId,
    pub contributing_nodes: Vec<NodeId>,
    pub detection_time: PtpTimestamp,
    pub confidence: f32,
    pub location_estimate: Option<LocationEstimate>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Location estimate from multi-node triangulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationEstimate {
    pub x: f32,  // meters
    pub y: f32,  // meters
    pub z: f32,  // meters
    pub confidence: f32,
}

/// Cluster configuration
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub cluster_id: ClusterId,
    pub tdma_frame_duration_ms: u64,
    pub tdma_guard_time_ms: u64,
    pub election_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
    pub max_nodes: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            cluster_id: "default_cluster".to_string(),
            tdma_frame_duration_ms: 100,
            tdma_guard_time_ms: 5,
            election_timeout_ms: 5000,
            heartbeat_interval_ms: 1000,
            max_nodes: 16,
        }
    }
}

/// Multi-node coordinator for cluster management
pub struct MultiNodeCoordinator {
    node_id: NodeId,
    config: ClusterConfig,
    nodes: Arc<Mutex<HashMap<NodeId, NodeInfo>>>,
    is_grandmaster: Arc<Mutex<bool>>,
    tdma_schedule: Arc<Mutex<Option<TdmaSchedule>>>,
    last_election: Arc<Mutex<Option<Instant>>>,
    election_count: Arc<Mutex<u32>>,
}

impl MultiNodeCoordinator {
    /// Create a new multi-node coordinator
    pub fn new(node_id: NodeId, config: ClusterConfig) -> Self {
        Self {
            node_id,
            config,
            nodes: Arc::new(Mutex::new(HashMap::new())),
            is_grandmaster: Arc::new(Mutex::new(false)),
            tdma_schedule: Arc::new(Mutex::new(None)),
            last_election: Arc::new(Mutex::new(None)),
            election_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get node ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get cluster ID
    pub fn cluster_id(&self) -> &str {
        &self.config.cluster_id
    }

    /// Check if this node is the grandmaster
    pub async fn is_grandmaster(&self) -> bool {
        *self.is_grandmaster.lock().await
    }

    /// Get election count
    pub async fn election_count(&self) -> u32 {
        *self.election_count.lock().await
    }

    /// Get known nodes
    pub async fn get_nodes(&self) -> Vec<NodeInfo> {
        self.nodes.lock().await.values().cloned().collect()
    }

    /// Get node count
    pub async fn node_count(&self) -> usize {
        self.nodes.lock().await.len()
    }

    /// Add or update a node in the cluster
    pub async fn update_node(&self, node_info: NodeInfo) -> Result<()> {
        let mut nodes = self.nodes.lock().await;

        if nodes.len() >= self.config.max_nodes && !nodes.contains_key(&node_info.node_id) {
            return Err(anyhow::anyhow!("Cluster at maximum capacity"));
        }

        nodes.insert(node_info.node_id.clone(), node_info);
        Ok(())
    }

    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: &str) -> bool {
        self.nodes.lock().await.remove(node_id).is_some()
    }

    /// Participate in grandmaster election
    pub async fn elect_grandmaster(&self, my_info: NodeInfo) -> ElectionResult {
        let mut nodes = self.nodes.lock().await;
        nodes.insert(my_info.node_id.clone(), my_info.clone());

        // Find best candidate based on:
        // 1. Clock class (lower is better)
        // 2. Clock accuracy (lower is better)
        // 3. Priority (lower is better)
        // 4. Node ID (lexicographic for tiebreaker)
        let best_node = nodes.values().min_by(|a, b| {
            (
                a.clock_class as u8,
                a.clock_accuracy as u8,
                a.priority,
                &a.node_id,
            )
                .cmp(&(
                    b.clock_class as u8,
                    b.clock_accuracy as u8,
                    b.priority,
                    &b.node_id,
                ))
        });

        match best_node {
            Some(node) if node.node_id == self.node_id => {
                *self.is_grandmaster.lock().await = true;
                *self.last_election.lock().await = Some(Instant::now());
                *self.election_count.lock().await += 1;
                ElectionResult::ElectedGrandmaster
            }
            Some(_) => {
                *self.is_grandmaster.lock().await = false;
                ElectionResult::NotElected
            }
            None => {
                // Should not happen since we inserted ourselves
                *self.is_grandmaster.lock().await = true;
                *self.last_election.lock().await = Some(Instant::now());
                *self.election_count.lock().await += 1;
                ElectionResult::ElectedGrandmaster
            }
        }
    }

    /// Generate TDMA schedule (only grandmaster should do this)
    pub async fn generate_tdma_schedule(&self) -> Result<TdmaSchedule> {
        if !self.is_grandmaster().await {
            return Err(anyhow::anyhow!("Only grandmaster can generate TDMA schedule"));
        }

        let nodes = self.nodes.lock().await;
        let node_list: Vec<_> = nodes.values().collect();
        let frame_duration_us = self.config.tdma_frame_duration_ms * 1000;
        let guard_time_us = self.config.tdma_guard_time_ms * 1000;

        let slot_duration_us = if node_list.is_empty() {
            frame_duration_us
        } else {
            (frame_duration_us - (guard_time_us * node_list.len() as u64)) / node_list.len() as u64
        };

        let mut slots = Vec::new();
        let mut current_time: u64 = (PtpTimestamp::from(chrono::Utc::now()).as_nanos() / 1000) as u64;

        for (index, node) in node_list.iter().enumerate() {
            slots.push(TdmaSlot {
                node_id: node.node_id.clone(),
                slot_index: index as u32,
                start_time_us: current_time,
                duration_us: slot_duration_us,
                guard_time_us,
            });
            current_time = current_time.saturating_add(slot_duration_us + guard_time_us);
        }

        let schedule = TdmaSchedule {
            slots,
            frame_duration_us,
            epoch: PtpTimestamp::from(chrono::Utc::now()),
        };

        *self.tdma_schedule.lock().await = Some(schedule.clone());
        Ok(schedule)
    }

    /// Get current TDMA schedule
    pub async fn get_tdma_schedule(&self) -> Option<TdmaSchedule> {
        self.tdma_schedule.lock().await.clone()
    }

    /// Get TDMA slot for this node
    pub async fn get_my_tdma_slot(&self) -> Option<TdmaSlot> {
        let schedule = self.get_tdma_schedule().await?;
        schedule.slots.into_iter().find(|slot| slot.node_id == self.node_id)
    }

    /// Fuse detection data from multiple nodes
    pub fn fuse_detection_data(
        &self,
        primary_data: FusedDetectionData,
        additional_data: Vec<FusedDetectionData>,
    ) -> Result<FusedDetectionData> {
        if additional_data.is_empty() {
            return Ok(primary_data);
        }

        let mut contributing_nodes = primary_data.contributing_nodes.clone();
        let mut all_confidences = vec![primary_data.confidence];

        for data in &additional_data {
            if !contributing_nodes.contains(&data.primary_node_id) {
                contributing_nodes.push(data.primary_node_id.clone());
            }
            all_confidences.push(data.confidence);
        }

        // Simple confidence fusion (weighted average)
        let avg_confidence: f32 = all_confidences.iter().sum::<f32>() / all_confidences.len() as f32;

        // Location fusion (simple average)
        let location_estimate = if primary_data.location_estimate.is_some()
            || additional_data.iter().any(|d| d.location_estimate.is_some())
        {
            let mut locations = Vec::new();
            if let Some(ref loc) = primary_data.location_estimate {
                locations.push(loc.clone());
            }
            for data in &additional_data {
                if let Some(ref loc) = data.location_estimate {
                    locations.push(loc.clone());
                }
            }
            if !locations.is_empty() {
                let n = locations.len() as f32;
                Some(LocationEstimate {
                    x: locations.iter().map(|l| l.x).sum::<f32>() / n,
                    y: locations.iter().map(|l| l.y).sum::<f32>() / n,
                    z: locations.iter().map(|l| l.z).sum::<f32>() / n,
                    confidence: avg_confidence,
                })
            } else {
                None
            }
        } else {
            None
        };

        Ok(FusedDetectionData {
            primary_node_id: primary_data.primary_node_id,
            contributing_nodes,
            detection_time: primary_data.detection_time,
            confidence: avg_confidence,
            location_estimate,
            metadata: primary_data.metadata,
        })
    }

    /// Check if election timeout has occurred
    pub async fn should_re_elect(&self) -> bool {
        if let Some(last) = *self.last_election.lock().await {
            last.elapsed() > Duration::from_millis(self.config.election_timeout_ms)
        } else {
            true
        }
    }

    /// Reset election timer
    pub async fn reset_election_timer(&self) {
        *self.last_election.lock().await = Some(Instant::now());
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node_info(id: &str, priority: u8) -> NodeInfo {
        NodeInfo {
            node_id: id.to_string(),
            cluster_id: "test_cluster".to_string(),
            clock_class: ClockClass::ClockClass248,
            clock_accuracy: ClockAccuracy::Nanosecond25,
            priority,
            last_seen: PtpTimestamp::from(chrono::Utc::now()),
            capabilities: NodeCapabilities {
                can_be_grandmaster: true,
                has_gps: false,
                has_atomic_clock: false,
                supports_tdma: true,
                max_sample_rate: 48000,
            },
        }
    }

    fn create_test_coordinator(node_id: &str) -> MultiNodeCoordinator {
        let config = ClusterConfig {
            cluster_id: "test_cluster".to_string(),
            max_nodes: 5,
            ..Default::default()
        };
        MultiNodeCoordinator::new(node_id.to_string(), config)
    }

    // ============================================================================
    // Basic Tests
    // ============================================================================

    #[tokio::test]
    async fn test_coordinator_creation() {
        let coordinator = create_test_coordinator("node1");
        assert_eq!(coordinator.node_id(), "node1");
        assert_eq!(coordinator.cluster_id(), "test_cluster");
        assert_eq!(coordinator.node_count().await, 0);
        assert!(!coordinator.is_grandmaster().await);
    }

    #[test]
    fn test_config_default() {
        let config = ClusterConfig::default();
        assert_eq!(config.cluster_id, "default_cluster");
        assert_eq!(config.tdma_frame_duration_ms, 100);
        assert_eq!(config.tdma_guard_time_ms, 5);
        assert_eq!(config.election_timeout_ms, 5000);
        assert_eq!(config.heartbeat_interval_ms, 1000);
        assert_eq!(config.max_nodes, 16);
    }

    // ============================================================================
    // Node Management Tests
    // ============================================================================

    #[tokio::test]
    async fn test_add_node() {
        let coordinator = create_test_coordinator("node1");
        let node_info = create_test_node_info("node2", 10);

        coordinator.update_node(node_info).await.unwrap();
        assert_eq!(coordinator.node_count().await, 1);
    }

    #[tokio::test]
    async fn test_add_node_at_capacity() {
        let coordinator = create_test_coordinator("node1");
        // Coordinator already has max_nodes=5

        // Add 5 other nodes (total capacity including self)
        for i in 1..=5 {
            let node_info = create_test_node_info(&format!("node{}", i + 1), 10);
            coordinator.update_node(node_info).await.unwrap();
        }

        // 6th node should fail
        let node_info = create_test_node_info("node7", 10);
        assert!(coordinator.update_node(node_info).await.is_err());
    }

    #[tokio::test]
    async fn test_remove_node() {
        let coordinator = create_test_coordinator("node1");

        coordinator.update_node(create_test_node_info("node2", 10)).await.unwrap();
        assert_eq!(coordinator.node_count().await, 1);

        assert!(coordinator.remove_node("node2").await);
        assert_eq!(coordinator.node_count().await, 0);
        assert!(!coordinator.remove_node("node2").await);  // Already removed
    }

    #[tokio::test]
    async fn test_get_nodes() {
        let coordinator = create_test_coordinator("node1");

        coordinator.update_node(create_test_node_info("node2", 10)).await.unwrap();
        coordinator.update_node(create_test_node_info("node3", 20)).await.unwrap();

        let nodes = coordinator.get_nodes().await;
        assert_eq!(nodes.len(), 2);
        assert!(nodes.iter().any(|n| n.node_id == "node2"));
        assert!(nodes.iter().any(|n| n.node_id == "node3"));
    }

    // ============================================================================
    // Grandmaster Election Tests
    // ============================================================================

    #[tokio::test]
    async fn test_elect_grandmaster_single_node() {
        let coordinator = create_test_coordinator("node1");
        let my_info = create_test_node_info("node1", 10);

        let result = coordinator.elect_grandmaster(my_info).await;
        assert_eq!(result, ElectionResult::ElectedGrandmaster);
        assert!(coordinator.is_grandmaster().await);
        assert_eq!(coordinator.election_count().await, 1);
    }

    #[tokio::test]
    async fn test_elect_grandmaster_by_priority() {
        let coordinator = create_test_coordinator("node1");

        // Add node with lower priority (higher number = lower priority)
        coordinator.update_node(create_test_node_info("node2", 20)).await.unwrap();

        let my_info = create_test_node_info("node1", 10);  // Higher priority
        let result = coordinator.elect_grandmaster(my_info).await;
        assert_eq!(result, ElectionResult::ElectedGrandmaster);
        assert!(coordinator.is_grandmaster().await);
    }

    #[tokio::test]
    async fn test_elect_grandmaster_lose_to_lower_priority() {
        let coordinator = create_test_coordinator("node1");

        // Add node with higher priority (lower number)
        coordinator.update_node(create_test_node_info("node2", 5)).await.unwrap();

        let my_info = create_test_node_info("node1", 10);  // Lower priority
        let result = coordinator.elect_grandmaster(my_info).await;
        assert_eq!(result, ElectionResult::NotElected);
        assert!(!coordinator.is_grandmaster().await);
    }

    #[tokio::test]
    async fn test_elect_grandmaster_by_clock_class() {
        let coordinator = create_test_coordinator("node1");

        // Create node with better clock class
        let mut better_node = create_test_node_info("node2", 10);
        better_node.clock_class = ClockClass::ClockClass6;  // Better than 248

        coordinator.update_node(better_node).await.unwrap();

        let my_info = create_test_node_info("node1", 10);
        let result = coordinator.elect_grandmaster(my_info).await;
        assert_eq!(result, ElectionResult::NotElected);
    }

    // ============================================================================
    // TDMA Schedule Tests
    // ============================================================================

    #[tokio::test]
    async fn test_generate_tdma_schedule() {
        let coordinator = create_test_coordinator("node1");

        // Set self as grandmaster
        coordinator.update_node(create_test_node_info("node1", 10)).await.unwrap();
        coordinator.update_node(create_test_node_info("node2", 20)).await.unwrap();
        coordinator.update_node(create_test_node_info("node3", 30)).await.unwrap();

        let result = coordinator.elect_grandmaster(create_test_node_info("node1", 10)).await;
        assert_eq!(result, ElectionResult::ElectedGrandmaster);

        let schedule = coordinator.generate_tdma_schedule().await.unwrap();
        assert_eq!(schedule.slots.len(), 3);
        assert_eq!(schedule.frame_duration_us, 100000);  // 100ms

        // Check that each node has a slot
        assert!(schedule.slots.iter().any(|s| s.node_id == "node1"));
        assert!(schedule.slots.iter().any(|s| s.node_id == "node2"));
        assert!(schedule.slots.iter().any(|s| s.node_id == "node3"));
    }

    #[tokio::test]
    async fn test_generate_tdma_schedule_not_grandmaster() {
        let coordinator = create_test_coordinator("node1");

        // Don't set self as grandmaster
        let result = coordinator.generate_tdma_schedule().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_my_tdma_slot() {
        let coordinator = create_test_coordinator("node1");

        coordinator.update_node(create_test_node_info("node1", 10)).await.unwrap();
        coordinator.update_node(create_test_node_info("node2", 20)).await.unwrap();

        coordinator.elect_grandmaster(create_test_node_info("node1", 10)).await;
        coordinator.generate_tdma_schedule().await.unwrap();

        let my_slot = coordinator.get_my_tdma_slot().await;
        assert!(my_slot.is_some());
        assert_eq!(my_slot.unwrap().node_id, "node1");
    }

    // ============================================================================
    // Data Fusion Tests
    // ============================================================================

    #[test]
    fn test_fuse_detection_data_single() {
        let coordinator = create_test_coordinator("node1");

        let data = FusedDetectionData {
            primary_node_id: "node1".to_string(),
            contributing_nodes: vec!["node1".to_string()],
            detection_time: PtpTimestamp::from(chrono::Utc::now()),
            confidence: 0.9,
            location_estimate: Some(LocationEstimate {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                confidence: 0.85,
            }),
            metadata: HashMap::new(),
        };

        let result = coordinator.fuse_detection_data(data, vec![]).unwrap();
        assert_eq!(result.confidence, 0.9);
        assert_eq!(result.contributing_nodes.len(), 1);
    }

    #[test]
    fn test_fuse_detection_data_multiple() {
        let coordinator = create_test_coordinator("node1");

        let primary = FusedDetectionData {
            primary_node_id: "node1".to_string(),
            contributing_nodes: vec!["node1".to_string()],
            detection_time: PtpTimestamp::from(chrono::Utc::now()),
            confidence: 0.9,
            location_estimate: Some(LocationEstimate {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                confidence: 0.85,
            }),
            metadata: HashMap::new(),
        };

        let secondary = FusedDetectionData {
            primary_node_id: "node2".to_string(),
            contributing_nodes: vec!["node2".to_string()],
            detection_time: PtpTimestamp::from(chrono::Utc::now()),
            confidence: 0.7,
            location_estimate: Some(LocationEstimate {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                confidence: 0.65,
            }),
            metadata: HashMap::new(),
        };

        let result = coordinator.fuse_detection_data(primary, vec![secondary]).unwrap();
        assert!((result.confidence - 0.8).abs() < 0.001);  // (0.9 + 0.7) / 2
        assert_eq!(result.contributing_nodes.len(), 2);
        assert!((result.location_estimate.unwrap().x - 1.5).abs() < 0.001);  // (1.0 + 2.0) / 2
    }

    // ============================================================================
    // Election Timer Tests
    // ============================================================================

    #[tokio::test]
    async fn test_should_re_elect_initially() {
        let coordinator = create_test_coordinator("node1");

        assert!(coordinator.should_re_elect().await);
    }

    #[tokio::test]
    async fn test_should_not_re_elect_recently() {
        let coordinator = create_test_coordinator("node1");

        coordinator.reset_election_timer().await;
        assert!(!coordinator.should_re_elect().await);
    }

    #[tokio::test]
    async fn test_reset_election_timer() {
        let coordinator = create_test_coordinator("node1");

        coordinator.elect_grandmaster(create_test_node_info("node1", 10)).await;
        let count = coordinator.election_count().await;

        coordinator.reset_election_timer().await;
        assert!(!coordinator.should_re_elect().await);
        assert_eq!(coordinator.election_count().await, count);  // Unchanged
    }

    // ============================================================================
    // Serialization Tests
    // ============================================================================

    #[test]
    fn test_node_capabilities_serialization() {
        let capabilities = NodeCapabilities {
            can_be_grandmaster: true,
            has_gps: false,
            has_atomic_clock: false,
            supports_tdma: true,
            max_sample_rate: 48000,
        };

        let json = serde_json::to_string(&capabilities).unwrap();
        let deserialized: NodeCapabilities = serde_json::from_str(&json).unwrap();

        assert!(deserialized.can_be_grandmaster);
        assert_eq!(deserialized.max_sample_rate, 48000);
    }

    #[test]
    fn test_location_estimate_serialization() {
        let location = LocationEstimate {
            x: 1.5,
            y: 2.5,
            z: 3.5,
            confidence: 0.9,
        };

        let json = serde_json::to_string(&location).unwrap();
        let deserialized: LocationEstimate = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.x, 1.5);
        assert_eq!(deserialized.confidence, 0.9);
    }

    #[test]
    fn test_tdma_slot_serialization() {
        let slot = TdmaSlot {
            node_id: "node1".to_string(),
            slot_index: 0,
            start_time_us: 1000,
            duration_us: 10000,
            guard_time_us: 100,
        };

        let json = serde_json::to_string(&slot).unwrap();
        let deserialized: TdmaSlot = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.node_id, "node1");
        assert_eq!(deserialized.slot_index, 0);
        assert_eq!(deserialized.duration_us, 10000);
    }
}

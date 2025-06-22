---------------------------------- MODULE ServiceDiscovery ----------------------------------
(* 
This is a TLA+ specification for the service discovery mechanism,
which is a key component of the microservice architecture.
It focuses on the service registry, health status tracking, and discovery operations.
*)

EXTENDS Naturals, FiniteSets, Sequences, TLC

\* Constants
CONSTANTS 
    Services,       \* Set of service definitions
    Instances,      \* Set of possible service instances
    Hosts,          \* Set of host machines
    Ports,          \* Set of available ports
    MaxInstances,   \* Maximum number of instances per service
    MaxImbalance    \* Maximum allowed load imbalance between instances

\* Variables
VARIABLES 
    registry,       \* Current service registry: mapping from service ID to set of instances
    healthStatus,   \* Health status of instances: mapping from instance ID to HEALTHY/UNHEALTHY
    requestCounts,  \* Count of requests sent to each instance: mapping from instance ID to count
    lastChecked     \* Last time each instance was checked: mapping from instance ID to time

vars == <<registry, healthStatus, requestCounts, lastChecked>>

\* Type definitions
InstanceID == [service: Services, id: Instances]
Instance == [
    id: InstanceID,
    host: Hosts,
    port: Ports,
    metadata: [STRING -> STRING]
]
HealthStatus == {"HEALTHY", "UNHEALTHY"}

\* Initial state
Init ==
    /\ registry = [s \in Services |-> {}]
    /\ healthStatus = [i \in {} |-> "HEALTHY"]
    /\ requestCounts = [i \in {} |-> 0]
    /\ lastChecked = [i \in {} |-> 0]

\* Helper functions
InstancesOf(service) == registry[service]

AllInstances == UNION {registry[s] : s \in Services}

HealthyInstances(service) ==
    {i \in registry[service] : healthStatus[i.id] = "HEALTHY"}

\* Register a new service instance
Register(service, instance) ==
    /\ instance.id.service = service
    /\ Cardinality(registry[service]) < MaxInstances
    /\ instance \notin registry[service]
    /\ registry' = [registry EXCEPT ![service] = registry[service] \cup {instance}]
    /\ healthStatus' = [healthStatus EXCEPT ![instance.id] = "HEALTHY"]
    /\ requestCounts' = [requestCounts EXCEPT ![instance.id] = 0]
    /\ lastChecked' = [lastChecked EXCEPT ![instance.id] = 0]

\* Deregister a service instance
Deregister(service, instance) ==
    /\ instance \in registry[service]
    /\ registry' = [registry EXCEPT ![service] = registry[service] \ {instance}]
    /\ healthStatus' = [s \in DOMAIN healthStatus \ {instance.id} |-> healthStatus[s]]
    /\ requestCounts' = [s \in DOMAIN requestCounts \ {instance.id} |-> requestCounts[s]]
    /\ lastChecked' = [s \in DOMAIN lastChecked \ {instance.id} |-> lastChecked[s]]

\* Update the health status of an instance
UpdateHealth(instance, status) ==
    /\ instance.id \in DOMAIN healthStatus
    /\ healthStatus' = [healthStatus EXCEPT ![instance.id] = status]
    /\ lastChecked' = [lastChecked EXCEPT ![instance.id] = lastChecked[instance.id] + 1]
    /\ UNCHANGED <<registry, requestCounts>>

\* Discover healthy instances of a service
Discover(service) ==
    LET discovered == HealthyInstances(service)
    IN discovered

\* Select an instance using load balancing
SelectInstance(service) ==
    LET 
        healthy == HealthyInstances(service)
        \* Find instance with minimum request count
        minRequests == 
            IF healthy = {} THEN 0
            ELSE CHOOSE min \in Nat : 
                /\ \E i \in healthy : requestCounts[i.id] = min
                /\ ~\E j \in healthy : requestCounts[j.id] < min
        candidateInstances == {i \in healthy : requestCounts[i.id] = minRequests}
    IN
    IF candidateInstances = {} THEN CHOOSE i \in {} : TRUE \* No instance available
    ELSE CHOOSE i \in candidateInstances : TRUE \* Select any candidate

\* Increment request count for an instance
IncrementRequestCount(instance) ==
    /\ instance.id \in DOMAIN requestCounts
    /\ requestCounts' = [requestCounts EXCEPT ![instance.id] = requestCounts[instance.id] + 1]
    /\ UNCHANGED <<registry, healthStatus, lastChecked>>

\* Perform health check for all instances
PerformHealthChecks ==
    \E status_fn \in [AllInstances -> HealthStatus] :
        /\ \A i \in AllInstances :
            healthStatus' = [healthStatus EXCEPT ![i.id] = status_fn[i]]
        /\ lastChecked' = [i \in DOMAIN lastChecked |-> lastChecked[i] + 1]
        /\ UNCHANGED <<registry, requestCounts>>

\* Complete service discovery process with load balancing
DiscoverAndSelect(service) ==
    /\ LET 
        instance == SelectInstance(service)
       IN
        /\ instance.id \in DOMAIN requestCounts
        /\ requestCounts' = [requestCounts EXCEPT ![instance.id] = requestCounts[instance.id] + 1]
    /\ UNCHANGED <<registry, healthStatus, lastChecked>>

\* System actions
Next ==
    \/ \E s \in Services, i \in Instance : Register(s, i)
    \/ \E s \in Services, i \in registry[s] : Deregister(s, i)
    \/ \E i \in AllInstances, status \in HealthStatus : UpdateHealth(i, status)
    \/ \E s \in Services : DiscoverAndSelect(s)
    \/ PerformHealthChecks

\* System specification
Spec == Init /\ [][Next]_vars

\* Invariants and properties

\* Type invariant: all registered instances should have health status, request count, etc.
TypeInvariant ==
    /\ \A s \in Services : \A i \in registry[s] : 
        /\ i.id \in DOMAIN healthStatus
        /\ i.id \in DOMAIN requestCounts
        /\ i.id \in DOMAIN lastChecked
    /\ \A id \in DOMAIN healthStatus : \E s \in Services : \E i \in registry[s] : i.id = id
    /\ \A id \in DOMAIN requestCounts : \E s \in Services : \E i \in registry[s] : i.id = id
    /\ \A id \in DOMAIN lastChecked : \E s \in Services : \E i \in registry[s] : i.id = id

\* No duplicate instances
NoDuplicateInstances ==
    \A s1, s2 \in Services :
        \A i1 \in registry[s1] :
            \A i2 \in registry[s2] :
                (s1 # s2 \/ i1.id # i2.id) => i1 # i2

\* Discovery consistency: All discovered instances must be healthy
ServiceDiscoveryConsistency ==
    \A s \in Services : \A i \in Discover(s) : healthStatus[i.id] = "HEALTHY"

\* Discovery availability: If healthy instances exist, discovery must return them
ServiceDiscoveryAvailability ==
    \A s \in Services : HealthyInstances(s) # {} => Discover(s) # {}

\* Load balancing: Request imbalance stays within bounds
LoadBalancingFairness ==
    \A s \in Services : \A i1, i2 \in HealthyInstances(s) :
        requestCounts[i1.id] - requestCounts[i2.id] <= MaxImbalance

\* Eventually, all unhealthy instances should not receive requests
EventuallyUnhealthyInstancesExcluded ==
    \A i \in AllInstances :
        []<>(healthStatus[i.id] = "UNHEALTHY" => 
            \A s \in Services : i \notin Discover(s))

\* Health checks update lastChecked timestamps
HealthChecksUpdateTimestamps ==
    [][PerformHealthChecks => \A i \in AllInstances : lastChecked'[i.id] > lastChecked[i.id]]_vars

================================================================================= 
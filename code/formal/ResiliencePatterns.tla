-------------------------------- MODULE ResiliencePatterns --------------------------------
(* 
This is a TLA+ specification for resilience patterns in microservice architectures.
It focuses on fault injection and resilience mechanisms including:
1. Circuit Breakers
2. Retries with Exponential Backoff
3. Fallbacks
4. Timeouts
5. Bulkheads (Isolation)

This specification extends the MicroserviceArchitecture module to add resilience testing.
*)

EXTENDS Naturals, FiniteSets, Sequences, TLC, MicroserviceArchitecture

CONSTANTS 
    MaxRetries,        \* Maximum retries allowed for a request
    CircuitTimeout,    \* Time in ticks when circuit breaker opens/closes
    MaxFailures,       \* Maximum failures before circuit breaker opens
    MaxLatency         \* Maximum acceptable latency

VARIABLES 
    circuitState,      \* Circuit breaker state for each instance: [CLOSED, OPEN, HALF_OPEN]
    failureCount,      \* Count of failures per service instance
    retryCount,        \* Retry count for each message
    elapsedTime,       \* Elapsed time for processing/timeouts
    pendingRequests,   \* Set of IDs of requests waiting for response
    latencies          \* Map from message ID to its latency

\* Resilience patterns vars
resilienceVars == <<circuitState, failureCount, retryCount, elapsedTime, pendingRequests, latencies>>

\* All variables
allVars == <<vars, resilienceVars>>

\* Type definitions
CircuitState == {"CLOSED", "OPEN", "HALF_OPEN"}

\* Initial state extension for resilience variables
ResilienceInit ==
    /\ circuitState = [i \in {} |-> "CLOSED"]
    /\ failureCount = [i \in {} |-> 0]
    /\ retryCount = [id \in {} |-> 0]
    /\ elapsedTime = 0
    /\ pendingRequests = {}
    /\ latencies = [id \in {} |-> 0]

\* Combined initial state
InitWithResilience ==
    /\ Init
    /\ ResilienceInit

\* Helper functions for resilience
ActiveCircuitBreaker(instance) ==
    /\ instance.id \in DOMAIN circuitState
    /\ circuitState[instance.id] # "CLOSED"

CanProcessRequest(instance) ==
    /\ instance.id \in DOMAIN circuitState
    /\ circuitState[instance.id] # "OPEN"
    
IsTimedOut(msgId) ==
    /\ msgId \in DOMAIN latencies
    /\ latencies[msgId] > MaxLatency

\* Actions for resilience patterns

\* 1. Circuit Breaker Actions

\* Open circuit breaker when failures exceed threshold
OpenCircuitBreaker(instance) ==
    /\ instance.id \in DOMAIN failureCount
    /\ instance.id \in DOMAIN circuitState
    /\ failureCount[instance.id] >= MaxFailures
    /\ circuitState[instance.id] = "CLOSED"
    /\ circuitState' = [circuitState EXCEPT ![instance.id] = "OPEN"]
    /\ UNCHANGED <<vars, failureCount, retryCount, elapsedTime, pendingRequests, latencies>>

\* Close circuit breaker after timeout with no failures
CloseCircuitBreaker(instance) ==
    /\ instance.id \in DOMAIN circuitState
    /\ circuitState[instance.id] = "HALF_OPEN"
    /\ elapsedTime >= CircuitTimeout
    /\ circuitState' = [circuitState EXCEPT ![instance.id] = "CLOSED"]
    /\ UNCHANGED <<vars, failureCount, retryCount, elapsedTime, pendingRequests, latencies>>

\* Try half-open state after timeout
TryHalfOpenCircuit(instance) ==
    /\ instance.id \in DOMAIN circuitState
    /\ circuitState[instance.id] = "OPEN"
    /\ elapsedTime >= CircuitTimeout
    /\ circuitState' = [circuitState EXCEPT ![instance.id] = "HALF_OPEN"]
    /\ UNCHANGED <<vars, failureCount, retryCount, elapsedTime, pendingRequests, latencies>>

\* 2. Retry with Exponential Backoff

\* Retry a failed request
RetryRequest(msg) ==
    /\ msg.id \in DOMAIN retryCount
    /\ retryCount[msg.id] < MaxRetries
    /\ msg.to \in DOMAIN healthStatus
    /\ \/ healthStatus[msg.to] = "UNHEALTHY"
       \/ ActiveCircuitBreaker({id |-> msg.to})
    /\ retryCount' = [retryCount EXCEPT ![msg.id] = retryCount[msg.id] + 1]
    /\ LET newMsg == [
            type |-> msg.type,
            from |-> msg.from,
            to |-> msg.to,
            content |-> msg.content,
            id |-> Len(messages) + 1
        ]
       IN messages' = Append(messages, newMsg)
    /\ pendingRequests' = pendingRequests \cup {newMsg.id}
    /\ latencies' = [latencies EXCEPT ![newMsg.id] = 0]
    /\ UNCHANGED <<registry, healthStatus, serviceState, circuitState, failureCount, elapsedTime>>

\* 3. Fallback Mechanism

\* Use fallback when service is unavailable
UseFallback(msg, fallback_service) ==
    /\ msg.id \in DOMAIN retryCount
    /\ retryCount[msg.id] >= MaxRetries
    /\ \E fallback_instance \in HealthyInstances(fallback_service) :
        /\ ~ActiveCircuitBreaker(fallback_instance)
        /\ LET fallbackMsg == [
                type |-> msg.type,
                from |-> msg.from,
                to |-> fallback_instance.id,
                content |-> "fallback_" \o msg.content,
                id |-> Len(messages) + 1
            ]
           IN messages' = Append(messages, fallbackMsg)
        /\ pendingRequests' = pendingRequests \cup {fallbackMsg.id}
        /\ latencies' = [latencies EXCEPT ![fallbackMsg.id] = 0]
    /\ UNCHANGED <<registry, healthStatus, serviceState, circuitState, failureCount, retryCount, elapsedTime>>

\* 4. Timeout Handling

\* Tick - increment elapsed time and message latencies
Tick ==
    /\ elapsedTime' = elapsedTime + 1
    /\ latencies' = [id \in DOMAIN latencies |-> latencies[id] + 1]
    /\ UNCHANGED <<vars, circuitState, failureCount, retryCount, pendingRequests>>

\* Handle timeout for a message
HandleTimeout ==
    /\ \E msgId \in pendingRequests :
        /\ IsTimedOut(msgId)
        /\ pendingRequests' = pendingRequests \ {msgId}
        /\ LET 
             originalMsg == CHOOSE msg \in messages : msg.id = msgId
             targetId == originalMsg.to
           IN
             /\ failureCount' = [failureCount EXCEPT ![targetId] = failureCount[targetId] + 1]
    /\ UNCHANGED <<vars, circuitState, retryCount, elapsedTime, latencies>>

\* 5. Bulkhead (Isolation) Pattern

\* Record success/failure for bulkhead monitoring
RecordSuccess(instance) ==
    /\ instance.id \in DOMAIN failureCount
    /\ failureCount' = [failureCount EXCEPT ![instance.id] = 0]
    /\ UNCHANGED <<vars, circuitState, retryCount, elapsedTime, pendingRequests, latencies>>

RecordFailure(instance) ==
    /\ instance.id \in DOMAIN failureCount
    /\ failureCount' = [failureCount EXCEPT ![instance.id] = failureCount[instance.id] + 1]
    /\ UNCHANGED <<vars, circuitState, retryCount, elapsedTime, pendingRequests, latencies>>

\* Fault Injection Actions

\* Inject failure by marking a healthy instance as unhealthy
InjectFailure(instance) ==
    /\ instance.id \in DOMAIN healthStatus
    /\ healthStatus[instance.id] = "HEALTHY"
    /\ healthStatus' = [healthStatus EXCEPT ![instance.id] = "UNHEALTHY"]
    /\ UNCHANGED <<registry, messages, serviceState, circuitState, failureCount, retryCount, elapsedTime, pendingRequests, latencies>>

\* Inject slowness by increasing latency
InjectLatency(msgId, additionalLatency) ==
    /\ msgId \in DOMAIN latencies
    /\ latencies' = [latencies EXCEPT ![msgId] = latencies[msgId] + additionalLatency]
    /\ UNCHANGED <<vars, circuitState, failureCount, retryCount, elapsedTime, pendingRequests>>

\* Next state for resilience patterns
ResilienceNext ==
    \/ \E instance \in AllInstances : OpenCircuitBreaker(instance)
    \/ \E instance \in AllInstances : CloseCircuitBreaker(instance)
    \/ \E instance \in AllInstances : TryHalfOpenCircuit(instance)
    \/ \E msg \in messages : RetryRequest(msg)
    \/ \E msg \in messages, service \in Services : UseFallback(msg, service)
    \/ HandleTimeout
    \/ Tick
    \/ \E instance \in AllInstances : RecordSuccess(instance)
    \/ \E instance \in AllInstances : RecordFailure(instance)
    \/ \E instance \in AllInstances : InjectFailure(instance)
    \/ \E msgId \in DOMAIN latencies, latency \in 1..MaxLatency : InjectLatency(msgId, latency)

\* Combined next state
NextWithResilience ==
    \/ Next
    \/ ResilienceNext

\* System specification with resilience
SpecWithResilience ==
    InitWithResilience /\ [][NextWithResilience]_allVars

\* Invariants

\* Circuit breaker should open after MaxFailures consecutive failures
CircuitBreakerOpensAfterMaxFailures ==
    \A instance \in AllInstances :
        instance.id \in DOMAIN failureCount /\ 
        failureCount[instance.id] >= MaxFailures => 
        circuitState[instance.id] = "OPEN"

\* Circuit breaker prevents requests to failing services
CircuitBreakerPreventsRequests ==
    \A msg \in messages :
        msg.to \in DOMAIN circuitState /\ 
        circuitState[msg.to] = "OPEN" => 
        ~(\E msg2 \in messages : 
            msg2.type = "REQUEST" /\ msg2.to = msg.to /\ msg2.id > msg.id)

\* Properties to check

\* Resilient system eventually processes all requests
EventuallyProcessAllRequests ==
    <>(\A msgId \in pendingRequests : 
        \E msg \in messages : 
            msg.type = "RESPONSE" /\ 
            msg.to = (CHOOSE origMsg \in messages : origMsg.id = msgId).from)

\* System recovers after failures
SystemEventuallyRecovers ==
    []<>(\A s \in Services : HealthyInstances(s) # {} => Discover(s))

\* Circuit breakers eventually close after system stabilizes
CircuitBreakersEventuallyClose ==
    []((\A instance \in AllInstances : 
        instance.id \in DOMAIN failureCount /\ 
        failureCount[instance.id] = 0) =>
        <>(\A instance \in AllInstances : 
            instance.id \in DOMAIN circuitState => 
            circuitState[instance.id] = "CLOSED"))

============================================================================= 
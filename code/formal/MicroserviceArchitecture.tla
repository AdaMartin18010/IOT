-------------------------------- MODULE MicroserviceArchitecture --------------------------------
(* 
This is a TLA+ specification for the microservice architecture formal model.
It focuses on the service discovery and communication aspects of the architecture.
*)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS 
    Services,       \* Set of service definitions
    Instances,      \* Set of possible service instances
    Hosts,          \* Set of host machines
    Ports,          \* Set of available ports
    MaxInstances    \* Maximum number of instances per service

VARIABLES 
    registry,       \* Current service registry: mapping from service ID to set of instances
    healthStatus,   \* Health status of instances: mapping from instance ID to HEALTHY/UNHEALTHY
    messages,       \* Messages in transit: sequence of message records
    serviceState    \* State of each service instance: mapping from instance ID to state

vars == <<registry, healthStatus, messages, serviceState>>

\* Type definitions
InstanceID == [service: Services, id: Instances]
Instance == [
    id: InstanceID,
    host: Hosts,
    port: Ports
]
HealthStatus == {"HEALTHY", "UNHEALTHY"}
MessageType == {"REQUEST", "RESPONSE", "EVENT"}
Message == [
    type: MessageType,
    from: InstanceID,
    to: InstanceID,
    content: STRING,
    id: Nat
]

\* Initial state
Init ==
    /\ registry = [s \in Services |-> {}]
    /\ healthStatus = [i \in {} |-> "HEALTHY"]
    /\ messages = <<>>
    /\ serviceState = [i \in {} |-> "INITIAL"]

\* Helper functions
InstancesOf(service) == registry[service]

AllInstances == UNION {registry[s] : s \in Services}

HealthyInstances(service) ==
    {i \in registry[service] : healthStatus[i.id] = "HEALTHY"}

\* Actions

\* Register a new service instance
Register(service, instance) ==
    /\ Cardinality(registry[service]) < MaxInstances
    /\ instance.id.service = service
    /\ instance \notin registry[service]
    /\ registry' = [registry EXCEPT ![service] = registry[service] \cup {instance}]
    /\ healthStatus' = [healthStatus EXCEPT ![instance.id] = "HEALTHY"]
    /\ serviceState' = [serviceState EXCEPT ![instance.id] = "INITIAL"]
    /\ UNCHANGED messages

\* Deregister a service instance
Deregister(service, instance) ==
    /\ instance \in registry[service]
    /\ registry' = [registry EXCEPT ![service] = registry[service] \ {instance}]
    /\ healthStatus' = [s \in DOMAIN healthStatus \ {instance.id} |-> healthStatus[s]]
    /\ serviceState' = [s \in DOMAIN serviceState \ {instance.id} |-> serviceState[s]]
    /\ UNCHANGED messages

\* Change health status of an instance
ChangeHealth(instance, status) ==
    /\ instance.id \in DOMAIN healthStatus
    /\ healthStatus' = [healthStatus EXCEPT ![instance.id] = status]
    /\ UNCHANGED <<registry, messages, serviceState>>

\* Send a synchronous request message
SendRequest(from, to_service) ==
    /\ from.id \in DOMAIN healthStatus
    /\ healthStatus[from.id] = "HEALTHY"
    /\ \E to \in HealthyInstances(to_service) :
        /\ LET msg == [
                type |-> "REQUEST",
                from |-> from.id,
                to |-> to.id,
                content |-> "request content",
                id |-> Len(messages) + 1
            ]
           IN messages' = Append(messages, msg)
    /\ UNCHANGED <<registry, healthStatus, serviceState>>

\* Process a request and send response
ProcessRequest ==
    /\ messages # <<>>
    /\ LET msg == Head(messages)
       IN /\ msg.type = "REQUEST"
          /\ msg.to \in DOMAIN healthStatus
          /\ healthStatus[msg.to] = "HEALTHY"
          /\ LET response == [
                  type |-> "RESPONSE",
                  from |-> msg.to,
                  to |-> msg.from,
                  content |-> "response content",
                  id |-> Len(messages) + 1
              ]
             IN /\ messages' = Append(Tail(messages), response)
                /\ serviceState' = [serviceState EXCEPT ![msg.to] = "PROCESSED_REQUEST"]
    /\ UNCHANGED <<registry, healthStatus>>

\* Send an asynchronous event message
SendEvent(from, to_service) ==
    /\ from.id \in DOMAIN healthStatus
    /\ healthStatus[from.id] = "HEALTHY"
    /\ \E to \in registry[to_service] :
        /\ LET msg == [
                type |-> "EVENT",
                from |-> from.id,
                to |-> to.id,
                content |-> "event content",
                id |-> Len(messages) + 1
            ]
           IN messages' = Append(messages, msg)
    /\ UNCHANGED <<registry, healthStatus, serviceState>>

\* Process an event message
ProcessEvent ==
    /\ messages # <<>>
    /\ LET msg == Head(messages)
       IN /\ msg.type = "EVENT"
          /\ msg.to \in DOMAIN healthStatus
          /\ messages' = Tail(messages)
          /\ serviceState' = [serviceState EXCEPT ![msg.to] = "PROCESSED_EVENT"]
    /\ UNCHANGED <<registry, healthStatus>>

\* Service discovery action
Discover(service) ==
    LET discovered == HealthyInstances(service)
    IN discovered # {}

\* System actions
Next ==
    \/ \E s \in Services, i \in Instance : Register(s, i)
    \/ \E s \in Services, i \in registry[s] : Deregister(s, i)
    \/ \E i \in AllInstances, status \in HealthStatus : ChangeHealth(i, status)
    \/ \E from \in AllInstances, to_service \in Services : SendRequest(from, to_service)
    \/ ProcessRequest
    \/ \E from \in AllInstances, to_service \in Services : SendEvent(from, to_service)
    \/ ProcessEvent

\* System specification
Spec == Init /\ [][Next]_vars

\* Invariants

\* All registered instances should have a health status
TypeInvariant ==
    /\ \A s \in Services : \A i \in registry[s] : i.id \in DOMAIN healthStatus
    /\ \A id \in DOMAIN healthStatus : \E s \in Services : \E i \in registry[s] : i.id = id

\* No duplicate instances
NoDuplicateInstances ==
    \A s1, s2 \in Services :
        \A i1 \in registry[s1] :
            \A i2 \in registry[s2] :
                (s1 # s2 \/ i1 # i2) => i1.id # i2.id

\* Properties to check

\* Service discovery always finds healthy instances if they exist
ServiceDiscoveryWorks ==
    [][\A s \in Services : HealthyInstances(s) # {} => Discover(s)]_vars

\* Every request eventually gets a response
RequestsGetResponses ==
    \A i \in Nat :
        i < Len(messages) /\ messages[i].type = "REQUEST" =>
        <>((\E j \in Nat : j < Len(messages) /\ 
            messages[j].type = "RESPONSE" /\ 
            messages[j].to = messages[i].from /\ 
            messages[j].from = messages[i].to))

============================================================================= 
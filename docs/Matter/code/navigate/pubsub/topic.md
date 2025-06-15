# define topic namespaces

## 0. root namespace for software world

rt

## 1. domain namespaces

domain : name of the domain  
gzb ：gezhouba
sx : shan xia
identifier : the domain identifier  
[domain].[identifier]  
examples:  
rt.gzb.navl.1  
rt.gzb.navl.2  
navl : means short name of navigation lock  

## 2. internal or external  namespaces

[internal|external]  
examples:  
rt.gzb.navl.1.in  
rt.gzb.navl.2.ex  

## 3. identify function namespaces

[system|service|component]  
system :  specific defined functions set. sys is short name of the system.  
service : abstract composite function. svc is short name of the service.  
component : entity specific defined function. cpt is short name of the component.  
examples:  
internal service or system or component  
rt.gzb.navl.1.in.sys.cpt.>  
rt.gzb.navl.1.in.svc.>  

## 4. identify components namespaces  

 [server|driver]  
 server : functions aggregation software  
 driver : specific function software for some hardware  
examples:  
internal  component server or driver  
rt.gzb.navl.1.in.cpt.svr.>  
rt.gzb.navl.1.in.cpt.dvr.>

## 5. identify components typename  

 driver typename : [radar|relay|tripodhead]  
 server typename : [playscreen|platform|navlstatus|stopline|shipspeed|waterline]  

## 6. identify component location namespaces

 [up|down].[left|right].[top|bottom]  
if component is defined , location namespaces will be none.  
location namespaces would be common physical constraint.  
examples:  
internal  component server or driver location  
rt.gzb.navl.1.in.cpt.svr.led  
rt.gzb.navl.1.in.cpt.dvr.radar.down.right  

## 7. identify component software static and dynamic properties

 [config]   **key/value properties for watch or setting.**  
 [action]   **publish action message**  
 [event]    **publish event message**  
 [cmd]      **reply command for request**  
 [status]   **reply current status for requre**  
examples:  
internal  component server or driver location  
rt.gzb.navl.1.in.cpt.svr.playscreen.cmd  
rt.gzb.navl.1.in.cpt.svr.navlstatus.status  
rt.gzb.navl.1.in.cpt.dvr.radar.down.right.cmd  

#!/bin/bash
echo "FilterComponentsLogs"

components_sys_array=("\[Kd\:Navlock,Id\:\#1\]","\[Kd\:Navlock,Id\:\#2\]","\[Kd\:Navlock,Id\:\#3\]")
components_svr_array=("navlstatus" ,"stopline","playscreen")
components_dvr_array=("ShipSpeed","Sensors","relay","radar","web")

echo "系统: ${components_sys_array[*]}"
echo "  服务: ${components_svr_array[*]}"
echo "   驱动: ${components_dvr_array[*]}"
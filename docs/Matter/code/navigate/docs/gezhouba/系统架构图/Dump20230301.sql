-- MySQL dump 10.13  Distrib 8.0.31, for Win64 (x86_64)
--
-- Host: localhost    Database: navigation
-- ------------------------------------------------------
-- Server version	8.0.31

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Current Database: `navigation`
--

CREATE DATABASE /*!32312 IF NOT EXISTS*/ `navigation` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;

USE `navigation`;

--
-- Table structure for table `iot_navl_cpnt_runtime_histories`
--

DROP TABLE IF EXISTS `iot_navl_cpnt_runtime_histories`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `iot_navl_cpnt_runtime_histories` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `created_at` datetime(3) DEFAULT NULL,
  `updated_at` datetime(3) DEFAULT NULL,
  `deleted_at` datetime(3) DEFAULT NULL,
  `navlock_id` varchar(255) DEFAULT NULL COMMENT '''船闸标识''',
  `nav_status` varchar(255) DEFAULT NULL COMMENT '''通航状态-通航上下行 通航换向-上行换下行-下行换上行''',
  `navlock_status` varchar(255) DEFAULT NULL COMMENT '''船闸状态-上下行出入闸''',
  `schedule_id` varchar(255) DEFAULT NULL COMMENT '''闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成''',
  `server_name` varchar(255) DEFAULT NULL COMMENT '''服务名称-监测,禁停,测速''',
  `system_name` varchar(255) DEFAULT NULL COMMENT '''系统名称-1#-监测,2#-禁停,3#-测速''',
  `driver_name` varchar(255) DEFAULT NULL COMMENT '''程序驱动名称-1#-监测,2#-禁停-下左,3#-测速-上右''',
  `format_version` bigint unsigned DEFAULT NULL COMMENT '''信息格式的版本号>=10000是测试类信息''',
  `info_level` bigint unsigned DEFAULT NULL COMMENT '''信息等级-1-info,2-警告,3-告警,4-错误,等''',
  `info` json DEFAULT NULL COMMENT '''信息内容-json格式''',
  PRIMARY KEY (`id`),
  KEY `idx_iot_navl_cpnt_runtime_histories_deleted_at` (`deleted_at`),
  KEY `idx_navlockid_1` (`navlock_id`),
  KEY `idx_navstatus_1` (`nav_status`),
  KEY `idx_navlockstatus_1` (`navlock_status`),
  KEY `idx_scheduleid_1` (`schedule_id`),
  KEY `idx_server_1` (`server_name`),
  KEY `idx_component_1` (`system_name`,`driver_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `iot_navl_cpnt_runtime_statuses`
--

DROP TABLE IF EXISTS `iot_navl_cpnt_runtime_statuses`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `iot_navl_cpnt_runtime_statuses` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `created_at` datetime(3) DEFAULT NULL,
  `updated_at` datetime(3) DEFAULT NULL,
  `deleted_at` datetime(3) DEFAULT NULL,
  `navlock_id` varchar(255) DEFAULT NULL COMMENT '''船闸标识''',
  `nav_status` varchar(255) DEFAULT NULL COMMENT '''通航状态-通航上下行 通航换向-上行换下行-下行换上行''',
  `navlock_status` varchar(255) DEFAULT NULL COMMENT '''船闸状态-上下行出入闸''',
  `schedule_id` varchar(255) DEFAULT NULL COMMENT '''闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成''',
  `server_name` varchar(255) DEFAULT NULL COMMENT '''服务名称-监测,禁停,测速''',
  `system_name` varchar(255) DEFAULT NULL COMMENT '''系统名称-1#-监测,2#-禁停,3#-测速''',
  `driver_name` varchar(255) DEFAULT NULL COMMENT '''程序驱动名称-1#-监测,2#-禁停-下左,3#-测速-上右''',
  `format_version` bigint unsigned DEFAULT NULL COMMENT '''信息格式的版本号>=10000是测试类信息''',
  `info_level` bigint unsigned DEFAULT NULL COMMENT '''信息等级-1-info,2-警告,3-告警,4-错误,等''',
  `info` json DEFAULT NULL COMMENT '''信息内容-json格式''',
  PRIMARY KEY (`id`),
  KEY `idx_component_2` (`system_name`,`driver_name`),
  KEY `idx_iot_navl_cpnt_runtime_statuses_deleted_at` (`deleted_at`),
  KEY `idx_navlockid_2` (`navlock_id`),
  KEY `idx_navstatus_2` (`nav_status`),
  KEY `idx_navlockstatus_2` (`navlock_status`),
  KEY `idx_scheduleid_2` (`schedule_id`),
  KEY `idx_server_2` (`server_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `iot_navl_devices_runtime_infos`
--

DROP TABLE IF EXISTS `iot_navl_devices_runtime_infos`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `iot_navl_devices_runtime_infos` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `created_at` datetime(3) DEFAULT NULL,
  `updated_at` datetime(3) DEFAULT NULL,
  `deleted_at` datetime(3) DEFAULT NULL,
  `navlock_id` varchar(255) DEFAULT NULL COMMENT '''船闸标识''',
  `nav_status` varchar(255) DEFAULT NULL COMMENT '''通航状态-通航上下行 通航换向-上行换下行-下行换上行''',
  `navlock_status` varchar(255) DEFAULT NULL COMMENT '''船闸状态-上下行出入闸''',
  `schedule_id` varchar(255) DEFAULT NULL COMMENT '''闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成''',
  `device_tag` varchar(255) DEFAULT NULL COMMENT '''设备标签-[上游左雷达..]''',
  `format_version` bigint unsigned DEFAULT NULL COMMENT '''信息格式的版本号''',
  `info` json DEFAULT NULL COMMENT '''信息内容-json格式''',
  PRIMARY KEY (`id`),
  KEY `idx_navlockstatus_6` (`navlock_status`),
  KEY `idx_scheduleid_6` (`schedule_id`),
  KEY `idx_devicetag_6` (`device_tag`),
  KEY `idx_iot_navl_devices_runtime_infos_deleted_at` (`deleted_at`),
  KEY `idx_navlockid_6` (`navlock_id`),
  KEY `idx_navstatus_6` (`nav_status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `iot_navl_schedule_status_histories`
--

DROP TABLE IF EXISTS `iot_navl_schedule_status_histories`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `iot_navl_schedule_status_histories` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `created_at` datetime(3) DEFAULT NULL,
  `updated_at` datetime(3) DEFAULT NULL,
  `deleted_at` datetime(3) DEFAULT NULL,
  `navlock_id` varchar(255) DEFAULT NULL COMMENT '''船闸标识''',
  `nav_status` varchar(255) DEFAULT NULL COMMENT '''通航状态-通航上下行 通航换向-上行换下行-下行换上行''',
  `navlock_status` varchar(255) DEFAULT NULL COMMENT '''船闸状态-上下行出入闸''',
  `schedule_id` varchar(255) DEFAULT NULL COMMENT '''闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成''',
  `format_version` bigint unsigned DEFAULT NULL COMMENT '''信息格式的版本号''',
  `info` json DEFAULT NULL COMMENT '''信息内容-json格式''',
  PRIMARY KEY (`id`),
  KEY `idx_iot_navl_schedule_status_histories_deleted_at` (`deleted_at`),
  KEY `idx_navlockid_3` (`navlock_id`),
  KEY `idx_navstatus_3` (`nav_status`),
  KEY `idx_navlockstatus_3` (`navlock_status`),
  KEY `idx_scheduleid_3` (`schedule_id`)
) ENGINE=InnoDB AUTO_INCREMENT=743 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `iot_navl_schedule_statuses`
--

DROP TABLE IF EXISTS `iot_navl_schedule_statuses`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `iot_navl_schedule_statuses` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `created_at` datetime(3) DEFAULT NULL,
  `updated_at` datetime(3) DEFAULT NULL,
  `deleted_at` datetime(3) DEFAULT NULL,
  `navlock_id` varchar(255) DEFAULT NULL COMMENT '''船闸标识''',
  `nav_status` varchar(255) DEFAULT NULL COMMENT '''通航状态-通航上下行 通航换向-上行换下行-下行换上行''',
  `navlock_status` varchar(255) DEFAULT NULL COMMENT '''船闸状态-上下行出入闸''',
  `schedule_id` varchar(255) DEFAULT NULL COMMENT '''闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成''',
  `format_version` bigint unsigned DEFAULT NULL COMMENT '''信息格式的版本号''',
  `info` json DEFAULT NULL COMMENT '''信息内容-json格式''',
  PRIMARY KEY (`id`),
  KEY `idx_iot_navl_schedule_statuses_deleted_at` (`deleted_at`),
  KEY `idx_navlockid_4` (`navlock_id`),
  KEY `idx_navstatus_4` (`nav_status`),
  KEY `idx_navlockstatus_4` (`navlock_status`),
  KEY `idx_scheduleid_4` (`schedule_id`)
) ENGINE=InnoDB AUTO_INCREMENT=204 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `iot_navl_speed_infos`
--

DROP TABLE IF EXISTS `iot_navl_speed_infos`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `iot_navl_speed_infos` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `created_at` datetime(3) DEFAULT NULL,
  `updated_at` datetime(3) DEFAULT NULL,
  `deleted_at` datetime(3) DEFAULT NULL,
  `navlock_id` varchar(255) DEFAULT NULL COMMENT '''船闸标识''',
  `nav_status` varchar(255) DEFAULT NULL COMMENT '''通航状态-通航上下行 通航换向-上行换下行-下行换上行''',
  `navlock_status` varchar(255) DEFAULT NULL COMMENT '''船闸状态-上下行出入闸''',
  `schedule_id` varchar(255) DEFAULT NULL COMMENT '''闸次号-标识一次通船的通航上行或者下行-从进闸到出闸的完成''',
  `format_version` bigint unsigned DEFAULT NULL COMMENT '''信息格式的版本号''',
  `info` json DEFAULT NULL COMMENT '''信息内容-json格式''',
  PRIMARY KEY (`id`),
  KEY `idx_iot_navl_speed_infos_deleted_at` (`deleted_at`),
  KEY `idx_navlockid_5` (`navlock_id`),
  KEY `idx_navstatus_5` (`nav_status`),
  KEY `idx_navlockstatus_5` (`navlock_status`),
  KEY `idx_scheduleid_5` (`schedule_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `navl_gzb_stopline_warns`
--

DROP TABLE IF EXISTS `navl_gzb_stopline_warns`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `navl_gzb_stopline_warns` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `created_at` datetime(3) DEFAULT NULL,
  `updated_at` datetime(3) DEFAULT NULL,
  `deleted_at` datetime(3) DEFAULT NULL,
  `navlock_id` varchar(255) DEFAULT NULL COMMENT '''船闸标识-[船闸名称]''',
  `schedule_id` varchar(255) DEFAULT NULL COMMENT '''闸次-[闸次号]''',
  `schedule_status` varchar(255) DEFAULT NULL COMMENT '''船闸调度状态-[上下行出入闸]''',
  `device_tag` varchar(255) DEFAULT NULL COMMENT '''设备标识-[ip地址]''',
  `cross_location` varchar(256) DEFAULT NULL COMMENT '''越线位置-[上游,下游]''',
  `cross_level` varchar(256) DEFAULT NULL COMMENT '''越线等级-[警告,报警]''',
  `stopline_width` bigint DEFAULT NULL COMMENT '''禁停线绿色区域宽度-[CM]''',
  `cross_distance` bigint DEFAULT NULL COMMENT '''越线距离-[CM]''',
  PRIMARY KEY (`id`),
  KEY `idx_navl_gzb_stopline_warns_deleted_at` (`deleted_at`),
  KEY `idx_navlock_12` (`navlock_id`),
  KEY `idx_scheduleid_12` (`schedule_id`),
  KEY `idx_schedulestatus_12` (`schedule_status`),
  KEY `idx_devicetag_12` (`device_tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `shipspeed`
--

DROP TABLE IF EXISTS `shipspeed`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `shipspeed` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `created_at` datetime(3) DEFAULT NULL,
  `updated_at` datetime(3) DEFAULT NULL,
  `deleted_at` datetime(3) DEFAULT NULL,
  `navlock_id` varchar(255) DEFAULT NULL COMMENT '''船闸标识-[船闸名称]''',
  `schedule_id` varchar(255) DEFAULT NULL COMMENT '''闸次-[闸次号]''',
  `schedule_status` varchar(255) DEFAULT NULL COMMENT '''船闸调度状态-[上下行出入闸]''',
  `device_tag` varchar(255) DEFAULT NULL COMMENT '''设备标识-ip地址''',
  `warn` varchar(256) DEFAULT NULL COMMENT '''警告-[超速]''',
  `speed` float DEFAULT NULL COMMENT '''船速-[m/s]''',
  `radar_tag` varchar(256) DEFAULT NULL COMMENT '''雷达tag-[上右,下左]''',
  PRIMARY KEY (`id`),
  KEY `idx_navlock_11` (`navlock_id`),
  KEY `idx_scheduleid_11` (`schedule_id`),
  KEY `idx_schedulestatus_11` (`schedule_status`),
  KEY `idx_devicetag_11` (`device_tag`),
  KEY `idx_shipspeed_deleted_at` (`deleted_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `speedlimit`
--

DROP TABLE IF EXISTS `speedlimit`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `speedlimit` (
  `Sid` bigint unsigned NOT NULL AUTO_INCREMENT COMMENT '''序号-int''',
  `Time` datetime(3) DEFAULT NULL COMMENT '''发生时间-UTC''',
  `NavlockId` varchar(256) DEFAULT NULL COMMENT '''船闸名称-str''',
  `ScheduleId` varchar(256) DEFAULT NULL COMMENT '''闸次-str''',
  `ScheduleStatus` varchar(256) DEFAULT NULL COMMENT '''船闸调度状态-str''',
  `DeviceTag` varchar(256) DEFAULT NULL COMMENT '''设备标识IP-str''',
  `Warn` varchar(256) DEFAULT NULL COMMENT '''警告Flag-str''',
  `Speed` float DEFAULT NULL COMMENT '''船速(m/s)-str''',
  PRIMARY KEY (`Sid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping events for database 'navigation'
--

--
-- Dumping routines for database 'navigation'
--

--
-- Current Database: `gvaweb`
--

CREATE DATABASE /*!32312 IF NOT EXISTS*/ `gvaweb` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;

USE `gvaweb`;

--
-- Table structure for table `nav_animation`
--

DROP TABLE IF EXISTS `nav_animation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_animation` (
  `id` int NOT NULL AUTO_INCREMENT,
  `createdAt` datetime(3) DEFAULT NULL,
  `navLockId` bigint DEFAULT NULL,
  `scheduleId` varchar(256) DEFAULT NULL,
  `navLockStatus` bigint DEFAULT NULL,
  `gateLeftUpStatus` bigint DEFAULT NULL,
  `gateLeftDownStatus` bigint DEFAULT NULL,
  `gateRightDownStatus` bigint DEFAULT NULL,
  `gateRightUpStatus` bigint DEFAULT NULL,
  `stoplineUpStatus` bigint DEFAULT NULL,
  `stoplineUpWidth` bigint DEFAULT NULL,
  `stoplineUpWarn` bigint DEFAULT NULL,
  `stoplineUpDistance` double DEFAULT NULL,
  `stoplineDownStatus` bigint DEFAULT NULL,
  `stoplineDownWidth` bigint DEFAULT NULL,
  `stoplineDownWarn` bigint DEFAULT NULL,
  `stoplineDownDistance` double DEFAULT NULL,
  `shipLeftDistance` double DEFAULT NULL,
  `shipLeftSpeed` double DEFAULT NULL,
  `shipLeftWarn` bigint DEFAULT NULL,
  `shipRightDistance` double DEFAULT NULL,
  `shipRightSpeed` double DEFAULT NULL,
  `shipRightWarn` bigint DEFAULT NULL,
  `selfLefeUpState` bigint DEFAULT NULL,
  `selfLefeDownState` bigint DEFAULT NULL,
  `selfRightUpState` bigint DEFAULT NULL,
  `selfRightDownState` bigint DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_nar_navlid_sche` (`createdAt`,`navLockId`,`scheduleId`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `nav_animation_history`
--

DROP TABLE IF EXISTS `nav_animation_history`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_animation_history` (
  `id` int NOT NULL AUTO_INCREMENT,
  `createdAt` datetime(3) DEFAULT NULL,
  `navLockId` bigint DEFAULT NULL,
  `scheduleId` varchar(256) DEFAULT NULL,
  `navLockStatus` bigint DEFAULT NULL,
  `gateLeftUpStatus` bigint DEFAULT NULL,
  `gateLeftDownStatus` bigint DEFAULT NULL,
  `gateRightDownStatus` bigint DEFAULT NULL,
  `gateRightUpStatus` bigint DEFAULT NULL,
  `stoplineUpStatus` bigint DEFAULT NULL,
  `stoplineUpWidth` bigint DEFAULT NULL,
  `stoplineUpWarn` bigint DEFAULT NULL,
  `stoplineUpDistance` double DEFAULT NULL,
  `stoplineDownStatus` bigint DEFAULT NULL,
  `stoplineDownWidth` bigint DEFAULT NULL,
  `stoplineDownWarn` bigint DEFAULT NULL,
  `stoplineDownDistance` double DEFAULT NULL,
  `shipLeftDistance` double DEFAULT NULL,
  `shipLeftSpeed` double DEFAULT NULL,
  `shipLeftWarn` bigint DEFAULT NULL,
  `shipRightDistance` double DEFAULT NULL,
  `shipRightSpeed` double DEFAULT NULL,
  `shipRightWarn` bigint DEFAULT NULL,
  `selfLefeUpState` bigint DEFAULT NULL,
  `selfLefeDownState` bigint DEFAULT NULL,
  `selfRightUpState` bigint DEFAULT NULL,
  `selfRightDownState` bigint DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_nar_navlid_sche` (`createdAt`,`navLockId`,`scheduleId`),
  KEY `idx_nar_navlid_sche0` (`createdAt`,`navLockId`,`scheduleId`)
) ENGINE=InnoDB AUTO_INCREMENT=57 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `nav_led_setup`
--

DROP TABLE IF EXISTS `nav_led_setup`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_led_setup` (
  `id` int NOT NULL AUTO_INCREMENT,
  `sortId` bigint DEFAULT NULL,
  `ledText` varchar(256) DEFAULT NULL,
  `navLockId` bigint DEFAULT NULL,
  `remark` varchar(256) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `index_ledsort` (`sortId`,`ledText`,`navLockId`) USING BTREE,
  KEY `idx_led` (`sortId`,`ledText`,`navLockId`)
) ENGINE=InnoDB AUTO_INCREMENT=14 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `nav_overspeed_warn`
--

DROP TABLE IF EXISTS `nav_overspeed_warn`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_overspeed_warn` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `createdAt` datetime(3) DEFAULT NULL,
  `navLockId` bigint DEFAULT NULL,
  `scheduleId` varchar(256) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `warn` varchar(256) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `distance` double DEFAULT NULL,
  `speed` double DEFAULT NULL,
  `scheduleStatus` bigint DEFAULT NULL,
  `radarTag` varchar(256) COLLATE utf8mb4_general_ci DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_sr_navlid_sche` (`createdAt`,`navLockId`,`scheduleId`)
) ENGINE=InnoDB AUTO_INCREMENT=122 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `nav_shipspeed_setup`
--

DROP TABLE IF EXISTS `nav_shipspeed_setup`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_shipspeed_setup` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `navLockId` bigint DEFAULT NULL,
  `distance` double DEFAULT NULL,
  `speedMax` double DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `index_navdis` (`navLockId`,`distance`) USING BTREE,
  KEY `idx_navdis` (`navLockId`,`distance`),
  KEY `idx_sss_navdis` (`navLockId`,`distance`)
) ENGINE=InnoDB AUTO_INCREMENT=35 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `nav_speed_curve_data`
--

DROP TABLE IF EXISTS `nav_speed_curve_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_speed_curve_data` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `scheduleStartTime` datetime(3) DEFAULT NULL,
  `navLockId` bigint DEFAULT NULL,
  `scheduleId` varchar(256) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `speedMax` double DEFAULT NULL,
  `inSpeed` double DEFAULT NULL,
  `outSpeed` double DEFAULT NULL,
  `scheduleStatus` bigint DEFAULT NULL,
  `dataSeq` json DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_sc_navlid_sche` (`scheduleStartTime`,`navLockId`,`scheduleId`)
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `nav_speed_data`
--

DROP TABLE IF EXISTS `nav_speed_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_speed_data` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `scheduleStartTime` datetime DEFAULT NULL COMMENT '闸次开始时间',
  `navLockId` bigint DEFAULT NULL COMMENT '船闸名称',
  `scheduleId` varchar(191) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '闸次号',
  `speedMax` double DEFAULT NULL COMMENT '最大速度',
  `inSpeed` double DEFAULT NULL COMMENT '进闸速度',
  `outSpeed` double DEFAULT NULL COMMENT '出闸速度',
  `scheduleStatus` bigint DEFAULT NULL COMMENT '航向',
  `distanceSeq` varchar(1000) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '距离序列，例 10,20,30米 写入[10,20,30]',
  `speedSeq` varchar(1000) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '速度序列，例 1,2,3米/秒 写入[1,2,3] 与距离序列对应',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `nav_speed_statis`
--

DROP TABLE IF EXISTS `nav_speed_statis`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_speed_statis` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `scheduleStartTime` datetime(3) DEFAULT NULL,
  `navLockId` bigint DEFAULT NULL,
  `scheduleId` varchar(256) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `overSpeedCount` bigint DEFAULT NULL,
  `upInCount` bigint DEFAULT NULL,
  `upOutCount` bigint DEFAULT NULL,
  `downInCount` bigint DEFAULT NULL,
  `downOutCount` bigint DEFAULT NULL,
  `maxSpeed` double DEFAULT NULL,
  `upInSpeed` double DEFAULT NULL,
  `upOutSpeed` double DEFAULT NULL,
  `downInSpeed` double DEFAULT NULL,
  `downOutSpeed` double DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_nss_navlid_sche` (`scheduleStartTime`,`navLockId`,`scheduleId`)
) ENGINE=InnoDB AUTO_INCREMENT=40 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `nav_stopline_statis`
--

DROP TABLE IF EXISTS `nav_stopline_statis`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_stopline_statis` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `scheduleStartTime` datetime(3) DEFAULT NULL,
  `navLockId` bigint DEFAULT NULL,
  `scheduleId` varchar(256) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `crossCount` bigint DEFAULT NULL,
  `upCount` bigint DEFAULT NULL,
  `downCount` bigint DEFAULT NULL,
  `warnCount` bigint DEFAULT NULL,
  `alarmCount` bigint DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_sls_navlid_sche` (`scheduleStartTime`,`navLockId`,`scheduleId`)
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `nav_stopline_warn`
--

DROP TABLE IF EXISTS `nav_stopline_warn`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nav_stopline_warn` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `createdAt` datetime(3) DEFAULT NULL,
  `navLockId` bigint DEFAULT NULL COMMENT '''船闸标识''',
  `scheduleId` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '''闸次-[闸次号]''',
  `scheduleStatus` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '''船闸调度状态-[上下行出入闸]''',
  `deviceTag` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '''设备标识-[ip地址]''',
  `crossLocation` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '''越线位置-[上游,下游]''',
  `crossLevel` varchar(255) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '''越线等级-[警告,报警]''',
  `stoplineWidth` bigint DEFAULT NULL COMMENT '''禁停线绿色区域宽度-[CM]''',
  `crossDistance` bigint DEFAULT NULL COMMENT '''越线距离-[CM]''',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_slw_navlid_sche` (`createdAt`,`navLockId`,`scheduleId`,`scheduleStatus`)
) ENGINE=InnoDB AUTO_INCREMENT=1001 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping events for database 'gvaweb'
--

--
-- Dumping routines for database 'gvaweb'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-03-01 13:54:06

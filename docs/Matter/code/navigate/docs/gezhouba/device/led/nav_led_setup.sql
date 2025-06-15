/*
 Navicat Premium Data Transfer

 Source Server         : localhost
 Source Server Type    : MySQL
 Source Server Version : 80029
 Source Host           : localhost:3306
 Source Schema         : gvaweb

 Target Server Type    : MySQL
 Target Server Version : 80029
 File Encoding         : 65001

 Date: 19/09/2022 14:32:46
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for nav_led_setup
-- ----------------------------
DROP TABLE IF EXISTS `nav_led_setup`;
CREATE TABLE `nav_led_setup`  (
  `id` int(0) NOT NULL AUTO_INCREMENT,
  `sortId` int(0) NULL DEFAULT NULL COMMENT '排序标识',
  `ledText` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT 'led文字',
  `navLockId` int(0) NULL DEFAULT NULL COMMENT '船闸名称',
  `remark` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `index_ledsort`(`sortId`, `ledText`, `navLockId`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 14 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of nav_led_setup
-- ----------------------------
INSERT INTO `nav_led_setup` VALUES (1, 1, '超速', 1, NULL);
INSERT INTO `nav_led_setup` VALUES (2, 2, 'm/s', 1, '1111');
INSERT INTO `nav_led_setup` VALUES (7, 1, '超速', 2, NULL);
INSERT INTO `nav_led_setup` VALUES (8, 2, 'm/s', 2, NULL);
INSERT INTO `nav_led_setup` VALUES (9, 3, '注意安全', 2, NULL);
INSERT INTO `nav_led_setup` VALUES (10, 1, '超速', 3, NULL);
INSERT INTO `nav_led_setup` VALUES (11, 2, 'm/s', 3, NULL);
INSERT INTO `nav_led_setup` VALUES (12, 3, '注意安全', 3, NULL);
INSERT INTO `nav_led_setup` VALUES (13, 3, '注意安全', 1, '');

SET FOREIGN_KEY_CHECKS = 1;

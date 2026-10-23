/******************* (C) COPYRIGHT 2025 Masses-Chip ************************
* File Name          : COMMON.H
* Author             : MASSES CHIP
* Version            : V1.07
* Date               : 2026/03/02
********************************************************************************
 


*******************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include "zc.h"
#include "MASSESCHIP_DEFINE.H"

void SYSTEM_INIT(void);						//系统初始化
void TC1_INIT(void);					//TC1初始化
void TC1_Interrupt (void);				//TC1中断处理函数
void TC1_Work(void);					//TC1处理函数
ab8 flag;
#define b_TC1_Flag flag.oneBit.b0
#endif

/*********************************END OF FILE************************************/
// 完整的企业微信数据收集脚本
// 获取部门、用户、私聊数据、群聊数据并写入表格

class WeChatDataCollector {
  constructor() {
    this.accessToken = null;
    this.departments = [];
    this.users = [];
    this.privateChatData = [];
    this.groupChatData = [];
  }

  // 获取access_token
  async getAccessToken() {
    try {
      console.log('正在获取access_token...');
      const response = await fetch('https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=ww3c73189347476992&corpsecret=7GPTkahno_JD1wiCmEI89i9mTutgTQjdjBYMpj1EJek');
      const data = await response.json();
      
      if (data.errcode === 0) {
        this.accessToken = data.access_token;
        console.log('access_token获取成功');
        return true;
      } else {
        throw new Error(`获取access_token失败: ${data.errmsg}`);
      }
    } catch (error) {
      console.error('获取access_token时发生错误:', error);
      return false;
    }
  }

  // 获取所有子部门ID列表
  async getDepartments() {
    try {
      console.log('\n=== 获取部门列表 ===');
      const response = await fetch(`https://qyapi.weixin.qq.com/cgi-bin/department/simplelist?access_token=${this.accessToken}`);
      const data = await response.json();
      
      console.log('部门接口响应:', JSON.stringify(data, null, 2));
      
      if (data.errcode === 0) {
        // 检查不同的可能字段名
        this.departments = data.department || [];
        console.log(`获取到 ${this.departments.length} 个部门`);
        
        if (this.departments.length > 0) {
          this.departments.forEach(dept => {
            console.log(`部门ID: ${dept.id}, 部门名称: ${dept.name}`);
          });
        } else {
          console.log('没有获取到部门数据，尝试手动添加根部门');
          // 如果没有部门数据，手动添加根部门
          this.departments = [{ id: 1, name: "根部门" }];
          console.log('已添加根部门');
        }
        return true;
      } else {
        throw new Error(`获取部门列表失败: ${data.errmsg}`);
      }
    } catch (error) {
      console.error('获取部门列表时发生错误:', error);
      return false;
    }
  }

  // 获取部门成员详情
  async getDepartmentUsers(departmentId) {
    try {
      const response = await fetch(`https://qyapi.weixin.qq.com/cgi-bin/user/list?access_token=${this.accessToken}&department_id=${departmentId}`);
      const data = await response.json();
      
      console.log(`部门 ${departmentId} 用户接口响应:`, JSON.stringify(data, null, 2));
      
      if (data.errcode === 0) {
        return data.userlist || [];
      } else {
        console.error(`获取部门 ${departmentId} 成员失败: ${data.errmsg}`);
        return [];
      }
    } catch (error) {
      console.error(`获取部门 ${departmentId} 成员时发生错误:`, error);
      return [];
    }
  }

  // 递归获取所有子部门ID
  async getAllDepartmentIds(parentId = 1, allIds = []) {
    const response = await fetch(`https://qyapi.weixin.qq.com/cgi-bin/department/list?access_token=${this.accessToken}&id=${parentId}`);
    const data = await response.json();
    if (data.errcode === 0 && data.department) {
      for (const dept of data.department) {
        if (!allIds.includes(dept.id)) {
          allIds.push(dept.id);
          // 递归获取子部门
          await this.getAllDepartmentIds(dept.id, allIds);
        }
      }
    }
    return allIds;
  }

  // 获取部门详情（根据部门ID获取部门名称）
  async getDepartmentInfo(departmentId) {
    try {
      console.log(`正在获取部门 ${departmentId} 的详情...`);
      const response = await fetch(`https://qyapi.weixin.qq.com/cgi-bin/department/get?access_token=${this.accessToken}&id=${departmentId}`);
      const result = await response.json();
      
      if (result.errcode === 0) {
        // 根据实际接口返回结构，部门名称在 result.department.name 中
        const deptName = result.department?.name || `部门${departmentId}`;
        console.log(`✅ 部门 ${departmentId} 详情获取成功:`, deptName);
        return deptName;
      } else {
        console.error(`❌ 部门 ${departmentId} 详情获取失败:`, result.errmsg);
        return `部门${departmentId}`;
      }
    } catch (error) {
      console.error(`获取部门 ${departmentId} 详情时发生错误:`, error);
      return `部门${departmentId}`;
    }
  }

  // 获取所有用户（递归所有部门，去重处理）
  async getAllUsers() {
    try {
      console.log('\n=== 递归获取所有用户（去重处理） ===');
      this.users = [];
      const userMap = new Map(); // 用于去重，key为userid
      
      // 递归获取所有部门ID
      const allDeptIds = await this.getAllDepartmentIds(1, []);
      console.log(`递归获取到 ${allDeptIds.length} 个部门ID`);
      
      for (const deptId of allDeptIds) {
        const users = await this.getDepartmentUsers(deptId);
        console.log(`部门 ${deptId} 获取到 ${users.length} 个用户`);
        
        // 处理每个用户，去重并记录部门信息
        for (const user of users) {
          if (userMap.has(user.userid)) {
            // 用户已存在，只记录多部门信息，不更新主要部门信息
            const existingUser = userMap.get(user.userid);
            if (!existingUser.departments) {
              existingUser.departments = [existingUser.department_id];
            }
            if (!existingUser.departments.includes(deptId)) {
              existingUser.departments.push(deptId);
            }
            console.log(`用户 ${user.name || user.userid} 已存在，添加部门 ${deptId}，保持原部门信息: ${existingUser.department_id}`);
          } else {
            // 新用户，添加到Map中，记录第一次获取到的部门信息
            user.department_id = deptId;
            user.departments = [deptId];
            userMap.set(user.userid, user);
            console.log(`新增用户: ${user.name || user.userid}，首次部门: ${deptId}`);
          }
        }
        
        await this.sleep(500);
      }
      
      // 将Map转换为数组
      this.users = Array.from(userMap.values());
      console.log(`去重后总共获取到 ${this.users.length} 个用户`);
      
      // 输出去重统计
      const duplicateUsers = this.users.filter(user => user.departments && user.departments.length > 1);
      if (duplicateUsers.length > 0) {
        console.log(`发现 ${duplicateUsers.length} 个多部门用户:`);
        duplicateUsers.forEach(user => {
          console.log(`  ${user.name || user.userid}: 部门 ${user.departments.join(', ')}`);
        });
      }
      
      return true;
    } catch (error) {
      console.error('递归获取所有用户时发生错误:', error);
      return false;
    }
  }

  // 获取私聊数据
  async getPrivateChatData(userIds, startTime, endTime) {
    try {
      console.log(`\n=== 获取私聊数据 (${userIds.length} 个用户) ===`);
      
      const requestData = {
        userid: userIds,
        start_time: startTime,
        end_time: endTime
      };

      const response = await fetch(`https://qyapi.weixin.qq.com/cgi-bin/externalcontact/get_user_behavior_data?access_token=${this.accessToken}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });

      const data = await response.json();
      
      if (data.errcode === 0) {
        console.log(`成功获取 ${data.behavior_data?.length || 0} 条私聊数据`);
        return data.behavior_data || [];
      } else {
        console.error(`获取私聊数据失败: ${data.errmsg}`);
        return [];
      }
    } catch (error) {
      console.error('获取私聊数据时发生错误:', error);
      return [];
    }
  }

  // 获取群聊数据
  async getGroupChatData(userIds, dayBeginTime) {
    try {
      console.log(`\n=== 获取群聊数据 (${userIds.length} 个用户) ===`);
      
      const requestData = {
        day_begin_time: dayBeginTime,
        owner_filter: {
          userid_list: userIds
        }
      };

      const response = await fetch(`https://qyapi.weixin.qq.com/cgi-bin/externalcontact/groupchat/statistic_group_by_day?access_token=${this.accessToken}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });

      const data = await response.json();
      
      if (data.errcode === 0) {
        console.log(`成功获取 ${data.items?.length || 0} 条群聊数据`);
        return data.items || [];
      } else {
        console.error(`获取群聊数据失败: ${data.errmsg}`);
        return [];
      }
    } catch (error) {
      console.error('获取群聊数据时发生错误:', error);
      return [];
    }
  }

  // 写入表格数据
  async writeToTable(data) {
    try {
      console.log('\n=== 写入表格数据 ===');
      console.log('准备写入数据条数:', data.length);
      
      // 验证数据格式
      if (!Array.isArray(data) || data.length === 0) {
        console.log('没有数据需要写入');
        return true;
      }
      
      // 逐条写入，确保values是对象而不是数组
      let successCount = 0;
      let failCount = 0;
      
      for (let i = 0; i < data.length; i++) {
        const singleData = data[i];
        console.log(`写入第 ${i + 1} 条数据`);
        console.log(`写入数据内容:`, JSON.stringify(singleData, null, 2));
        console.log(`平均首次回复时长值:`, singleData["平均首次回复时长"]);
        
        const requestData = {
          action: "通用写入表单",
          company: "家庭医生",
          WordList: {
            docid: "dcOKLm63xOSYsTmzdCFmGJehWl6idBXDW-h6WjvxA_xyQ0k7pNfpA4OsnRRiCxtHkQ_tHa_LhTHiRjNrQj_1Al4w",
            sheet_id: "tgOTx2",
            view_id: "vFlmaw",
            values: singleData
          }
        };

        try {
          const response = await fetch('https://api.yxkf120.com/QYCurrencyProxySidebarAPI', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
          });

          const result = await response.json();
          console.log(`第 ${i + 1} 条写入结果:`, result);
          
          if (result.success || result.errcode === 0) {
            console.log(`✅ 第 ${i + 1} 条数据写入成功`);
            successCount++;
          } else {
            console.error(`❌ 第 ${i + 1} 条数据写入失败:`, result.errmsg || result.message);
            failCount++;
          }
          
          // 避免请求过于频繁
          await this.sleep(1000);
          
        } catch (error) {
          console.error(`❌ 第 ${i + 1} 条数据写入异常:`, error);
          failCount++;
        }
      }
      
      console.log(`\n=== 写入完成 ===`);
      console.log(`成功写入: ${successCount} 条`);
      console.log(`失败写入: ${failCount} 条`);
      
      return failCount === 0;
    } catch (error) {
      console.error('写入表格时发生错误:', error);
      return false;
    }
  }

  // 处理数据并写入表格
  async processAndWriteData() {
    try {
      console.log('\n=== 开始处理数据 ===');
      
      // 设置时间范围（昨天）
      const now = Math.floor(Date.now() / 1000);
      const oneDay = 86400;
      const yesterday = now - oneDay;
      const yesterdayStart = Math.floor(yesterday / oneDay) * oneDay; // 昨天0点时间戳
      
      console.log(`处理时间范围: ${new Date(yesterdayStart * 1000).toISOString()} (昨天)`);
      
      // 逐个处理用户，确保每个用户获取到自己的数据
      const allData = [];
      
      for (let i = 0; i < this.users.length; i++) {
        const user = this.users[i];
        console.log(`\n=== 处理用户 ${i + 1}/${this.users.length}: ${user.name || user.userid} ===`);
        
        // 单独获取该用户的私聊数据
        const privateData = await this.getPrivateChatData([user.userid], yesterdayStart, yesterdayStart + oneDay);
        console.log(`用户 ${user.name || user.userid} 的私聊数据条数:`, privateData.length);
        if (privateData.length > 0) {
          console.log('私聊数据详情:', JSON.stringify(privateData, null, 2));
        }
        
        // 单独获取该用户的群聊数据（只有群主才能获取到数据）
        const groupData = await this.getGroupChatData([user.userid], yesterdayStart);
        console.log(`用户 ${user.name || user.userid} 的群聊数据条数:`, groupData.length);
        if (groupData.length > 0) {
          console.log('群聊数据详情:', JSON.stringify(groupData, null, 2));
        }
        
        // 处理该用户的数据
        console.log(`\n=== 处理用户数据: ${user.name || user.userid} ===`);
        
        const targetTimestamp = yesterdayStart;
        console.log('目标时间戳:', targetTimestamp);
        console.log('目标日期:', new Date(targetTimestamp * 1000).toISOString());
        
        // 查找对应日期的私聊数据
        let privateItem = privateData.find(item => {
          const itemDate = new Date(item.stat_time * 1000);
          const targetDateObj = new Date(targetTimestamp * 1000);
          const dateMatch = itemDate.getDate() === targetDateObj.getDate() && 
                           itemDate.getMonth() === targetDateObj.getMonth() && 
                           itemDate.getFullYear() === targetDateObj.getFullYear();
          
          console.log(`私聊数据检查 - 日期: ${itemDate.toISOString()}, 日期匹配: ${dateMatch}`);
          
          return dateMatch;
        });
        
        // 查找对应日期的群聊数据
        let groupItem = groupData.find(item => {
          const itemDate = new Date(item.stat_time * 1000);
          const targetDateObj = new Date(targetTimestamp * 1000);
          const dateMatch = itemDate.getDate() === targetDateObj.getDate() && 
                           itemDate.getMonth() === targetDateObj.getMonth() && 
                           itemDate.getFullYear() === targetDateObj.getFullYear();
          
          console.log(`群聊数据检查 - 日期: ${itemDate.toISOString()}, 日期匹配: ${dateMatch}`);
          
          return dateMatch;
        });
        
        // 如果没有找到对应日期的数据，使用所有数据
        if (!privateItem && privateData.length > 0) {
          privateItem = {
            chat_cnt: privateData.reduce((sum, item) => sum + (item.chat_cnt || 0), 0),
            message_cnt: privateData.reduce((sum, item) => sum + (item.message_cnt || 0), 0),
            negative_feedback_cnt: privateData.reduce((sum, item) => sum + (item.negative_feedback_cnt || 0), 0),
            new_apply_cnt: privateData.reduce((sum, item) => sum + (item.new_apply_cnt || 0), 0),
            new_contact_cnt: privateData.reduce((sum, item) => sum + (item.new_contact_cnt || 0), 0),
            reply_percentage: privateData.reduce((sum, item) => sum + (item.reply_percentage || 0), 0),
            avg_reply_time: privateData.reduce((sum, item) => sum + (item.avg_reply_time || 0), 0)
          };
          console.log('汇总私聊数据:', privateItem);
        } else {
          console.log(`用户 ${user.name || user.userid} 没有任何私聊数据`);
        }
        
        if (!groupItem && groupData.length > 0) {
          groupItem = {
            data: {
              new_chat_cnt: groupData.reduce((sum, item) => sum + (item.data?.new_chat_cnt || 0), 0),
              chat_total: groupData.reduce((sum, item) => sum + (item.data?.chat_total || 0), 0),
              chat_has_msg: groupData.reduce((sum, item) => sum + (item.data?.chat_has_msg || 0), 0),
              new_member_cnt: groupData.reduce((sum, item) => sum + (item.data?.new_member_cnt || 0), 0),
              member_total: groupData.reduce((sum, item) => sum + (item.data?.member_total || 0), 0),
              member_has_msg: groupData.reduce((sum, item) => sum + (item.data?.member_has_msg || 0), 0),
              msg_total: groupData.reduce((sum, item) => sum + (item.data?.msg_total || 0), 0)
            }
          };
          console.log('汇总群聊数据:', groupItem);
        } else {
          console.log(`用户 ${user.name || user.userid} 没有任何群聊数据（可能不是群主）`);
        }
        
        // 获取部门名称（处理多部门用户）
        let departmentName;
        if (user.departments && user.departments.length > 1) {
          // 多部门用户，获取所有部门名称
          const departmentNames = [];
          for (const deptId of user.departments) {
            const deptName = await this.getDepartmentInfo(deptId);
            departmentNames.push(deptName);
          }
          departmentName = departmentNames.join('、');
          console.log(`用户 ${user.name || user.userid} 多部门: ${departmentName}`);
        } else {
          // 单部门用户
          departmentName = await this.getDepartmentInfo(user.department_id);
        }
        
        // 确保所有数值都是数字类型，字段名完全匹配表结构
        const dataItem = {
          人员: [{
            user_id: user.userid || ""
          }],
          "删除/拉黑成员的客户数": Number(privateItem?.negative_feedback_cnt || 0),
          "发送消息数": Number(privateItem?.message_cnt || 0),
          "客户群新增群人数": Number(groupItem?.data?.new_member_cnt || 0),
          "已回复聊天占比": Number(privateItem?.reply_percentage || 0),
          "平均首次回复时长": [{
            "format": {
              "type": "text"
            },
            "text": String(privateItem?.avg_reply_time || 0),
            "type": "text"
          }],
          "截至当天客户群总人数": Number(groupItem?.data?.member_total || 0),
          "截至当天客户群总数量": Number(groupItem?.data?.chat_total || 0),
          "截至当天客户群消息总数": Number(groupItem?.data?.msg_total || 0),
          所属部门: [{
            format: {},
            text: departmentName,
            type: "text"
          }],
          "新增客户数": Number(privateItem?.new_contact_cnt || 0),
          "新增客户群数量": Number(groupItem?.data?.new_chat_cnt || 0),
          "聊天总数": Number(privateItem?.chat_cnt || 0),
          "记录日期": String(yesterdayStart * 1000)
        };
        
        console.log('最终构建的数据项:', JSON.stringify(dataItem, null, 2));
        
        // 检查是否所有数值字段都为0
        const numericFields = [
          "删除/拉黑成员的客户数",
          "发送消息数", 
          "客户群新增群人数",
          "已回复聊天占比",
          "截至当天客户群总人数",
          "截至当天客户群总数量", 
          "截至当天客户群消息总数",
          "新增客户数",
          "新增客户群数量",
          "聊天总数"
        ];
        
        const allZero = numericFields.every(field => dataItem[field] === 0);
        
        if (allZero) {
          console.log(`⚠️ 用户 ${user.name || user.userid} 的所有数据都为0，跳过写入`);
        } else {
          console.log(`✅ 用户 ${user.name || user.userid} 有有效数据，准备写入`);
          allData.push(dataItem);
        }
        
        // 避免请求过于频繁
        await this.sleep(1000);
      }
      
      console.log(`\n总共处理了 ${allData.length} 条有效数据（已过滤全为0的数据）`);
      
      // 写入表格
      if (allData.length > 0) {
        await this.writeToTable(allData);
      } else {
        console.log('⚠️ 没有有效数据需要写入表格');
      }
      
      return true;
    } catch (error) {
      console.error('处理数据时发生错误:', error);
      return false;
    }
  }

  // 主执行函数
  async run() {
    try {
      console.log('=== 开始执行企业微信数据收集 ===');
      
      // 1. 获取access_token
      if (!await this.getAccessToken()) {
        return false;
      }
      
      // 2. 获取部门列表
      if (!await this.getDepartments()) {
        return false;
      }
      
      // 3. 获取所有用户
      if (!await this.getAllUsers()) {
        return false;
      }
      
      // 4. 处理数据并写入表格
      if (!await this.processAndWriteData()) {
        return false;
      }
      
      console.log('\n=== 数据收集完成 ===');
      return true;
      
    } catch (error) {
      console.error('执行过程中发生错误:', error);
      return false;
    }
  }

  // 工具函数：延时
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// 定时运行函数
function scheduleNextRun() {
  const now = new Date();
  const tomorrow = new Date(now);
  tomorrow.setDate(tomorrow.getDate() + 1);
  tomorrow.setHours(0, 0, 0, 0); // 设置为明天0点
  
  const timeUntilNextRun = tomorrow.getTime() - now.getTime();
  
  console.log(`下次运行时间: ${tomorrow.toLocaleString()}`);
  console.log(`距离下次运行还有: ${Math.floor(timeUntilNextRun / 1000 / 60)} 分钟`);
  
  setTimeout(async () => {
    console.log('\n=== 定时任务开始执行 ===');
    const collector = new WeChatDataCollector();
    const success = await collector.run();
    
    if (success) {
      console.log('✅ 定时任务执行成功');
    } else {
      console.log('❌ 定时任务执行失败');
    }
    
    // 继续下一次定时
    scheduleNextRun();
  }, timeUntilNextRun);
}

// 立即运行一次
console.log('=== 启动企业微信数据收集定时任务 ===');
const collector = new WeChatDataCollector();
collector.run().then(success => {
  if (success) {
    console.log('✅ 首次运行成功');
  } else {
    console.log('❌ 首次运行失败');
  }
  
  // 设置定时任务
  scheduleNextRun();
}); 
# 这是一个自学Agent开发的学习过程，分为6步

## step1 跑通最简Agent

- 选择Pydantic AI，10行代码起步
- 接入一个搜索工具Tavily
- **过关标准** ：Agent自己搜资料+回答一个需要查证的问题


## step2 搭起完整的ReAct循环

- 升级到一个真正的编排框架：LangGraph
- 给Agent加3个工具：搜索/读文件/写文件
- 接LangSmith/Langfuse/Phoenix看完整Trace
- **过关标准** ：能让Agent完成(调研某话题->输出报告到本地)这类多步任务，且Trace能看懂


## step3 给Agent加上记忆

- 接Chroma/Qdrant当Vector DB
- 实现一个简单的Mem0风格记忆层(user_facts表+检索)
- 把对话历史压缩+索引
- **过关标准** ：第二次对话时Agent能记得你上次说过的关键信息


## step4 多Agent协作

- 用CrewAI或LangGraph实现一个3-Agent流水线（研究员+写手+编辑）
- 同一个任务用Sub-Agent模式再写一遍
- 比较两种模式在上下文消耗、可控性、调试难度上的差异
- **过关标准** ：能讲清楚什么时候该用Sub-Agent，什么时候该用Team


## step5 把Eval当一等公民

- 给你的Agent设计5条评估标准（成功率/调用次数/成本/延迟/用户满意度）
- 用**LangSmith Eval**或**Ragas**跑批 
- 找出失败case，做归因（Harness问题？模型问题？Tool问题？）
- **过关标准** ：能画出Agent的失败率分布图，并知道下一步该改哪里


## step6 上线+业务化

- 部署到Vercel/Railway/自己VPS（带监控）
- 找5个真实用户用一周，收集反馈
- 用Part6的7维框架评估你的项目
- **过关标准** ：清晰回答（这东西到底解决谁的什么问题，凭什么收钱）
# paie model
import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel
from utils import hungarian_matcher, get_best_span, get_best_span_simple


class PAIE(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)
        self.w_prompt_start = nn.Parameter(torch.rand(config.d_model, )) # dimension of bart : 768
        self.w_prompt_end = nn.Parameter(torch.rand(config.d_model, ))

       # self.model._init_weights(self.w_prompt_start) #似乎没用
       # self.model._init_weights(self.w_prompt_end)  #感觉没用，不是linear或 embedding的实例
        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')


    def forward(
        self,
        enc_input_ids=None,
        enc_mask_ids=None,
        dec_prompt_ids=None,
        dec_prompt_mask_ids=None,
        arg_joint_prompts=None,
        target_info=None,
        old_tok_to_new_tok_indexs=None,
        arg_list=None,
    ):
        """
        Args:
            multi args post calculation
        """
        if self.config.context_representation == 'decoder': # 如果选择的是decoder, 刚走全流程，这是上下文表示的原始语料
            context_outputs = self.model( #输入encoder->decoder->seq2seq 
                enc_input_ids,
                attention_mask=enc_mask_ids, # 批大小  32
                return_dict=True,
            )
            decoder_context = context_outputs.encoder_last_hidden_state
            context_outputs = context_outputs.last_hidden_state   #encoder-decoder后的上下文输出 信息
        else:
            context_outputs = self.model.encoder(
                enc_input_ids,
                attention_mask=enc_mask_ids,
            )
            context_outputs = context_outputs.last_hidden_state
            decoder_context = context_outputs

        decoder_prompt_outputs = self.model.decoder( #decoder中输入的是prompt信息
                input_ids=dec_prompt_ids,
                attention_mask=dec_prompt_mask_ids,
                encoder_hidden_states=decoder_context,
                encoder_attention_mask=enc_mask_ids,
        )
        decoder_prompt_outputs = decoder_prompt_outputs.last_hidden_state   #[bs, prompt_len, H]  #得到prompt 信息输入decoder后的输出 

        logit_lists = list()
        total_loss = 0.
        for i, (context_output, decoder_prompt_output, arg_joint_prompt, old_tok_to_new_tok_index) in \
            enumerate(zip(context_outputs, decoder_prompt_outputs, arg_joint_prompts, old_tok_to_new_tok_indexs)): # arg_joint_prompts 对事件模板中的每一种进行处理 ;  old_tok_to_new_tok_indexs 为模板中的偏移和bart转换后的各个槽位的偏移对照表,用于取出推理后的 向量。
            
            batch_loss = list()
            cnt = 0 #计数器？
            
            output = dict()
            for arg_role in arg_joint_prompt.keys(): # 对每个角色进行处理，比如place, target,attacker,instrument等
                """
                "arg_role": {"tok_s": , "tok_e": }
                """
                prompt_slots = arg_joint_prompt[arg_role] #得到各个角色的机槽位的偏移。 prompt_slots 中得到的是推理出来的槽位的起始地址，28，29； 31，32  这是两个mask的偏移。 问题： 这偏移是原句还是token id中的？

                start_logits_list = list()  # 开始位置的槽位的logits  ,用来算loss 
                end_logits_list = list()    #结束 位置的槽位的logits 
                for (p_start,p_end) in zip(prompt_slots['tok_s'], prompt_slots['tok_e']): #每个槽位的起止位置
                    prompt_query_sub = decoder_prompt_output[p_start:p_end] #用某槽位的一对开始结束偏移得到这个角色的开始结束地址的嵌入向量  形状为 1* 768 单个向量； 在bart输出向量中找到对应的嵌入值 ，用来算下面的其它值 （原材料）
                    prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0) # 如果有多个向量，求它的均值 作为表示。dim=0, 表示对最外围的向量求均值，因为是1*768， 那就是这个768维的向量自己和自己求均值 ，那就是本身了，目的是将一个名多个向量表示转为一个向量表示，好计算。 mean 会导致降维，unsqueeze 给它恢复维度。
                    
                    start_query = (prompt_query_sub*self.w_prompt_start).unsqueeze(-1) # [1* 768] X [1 * 768]    #将得到的向量与参数表相乘，等效于两个对角矩阵相乘的简化表示，本质就是线性映射 ，映射 到query 空间。 得到 1* 768*1
                    end_query = (prompt_query_sub*self.w_prompt_end).unsqueeze(-1)     # [1, H, 1]   得到 1* 768*1

                    start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()   #用 [ 1*180*768 ]形状的output向量 乘 [1*768 *1]  得到logitst，180个标量值 （跟序列长度相同个数）.得到在context空间的相似度值，其实就是从context中取要的位置的权重？
                    end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()  #结果也是  180 个标量值 ，跟序列长度相同。同上，处理的是end位置
                    
                    start_logits_list.append(start_logits)  #保存得到的start logits ，这是通过模板去 原文context中找出来的槽位偏移，并不一定是真实的，即预测值 
                    end_logits_list.append(end_logits) #end当前槽位的end位置的logits, 其它同上
                 #上面是把同一个角色的所有偏移都计算完成。   
                output[arg_role] = [start_logits_list, end_logits_list] # 跨度输出，原文是180字（默认），输出的就是首尾位置mask对应的logist值的数据对，每个的长度与序列长度相似。

                if self.training:  #计算损失
                    # calculate loss
                    target = target_info[i][arg_role] # "arg_role": {"text": ,"span_s": ,"span_e": }  找出样本中实际的当前角色的实际槽位及偏移。
                    predicted_spans = list()
                    for (start_logits, end_logits) in zip(start_logits_list, end_logits_list): # 找出上面找到的logists 对应的所有的start/end 对的值 
                        if self.config.matching_method_train == 'accurate':  #准确度优先？
                            predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index, self.config.max_span_length))
                        elif self.config.matching_method_train == 'max':  #匹配数优势（召回优先？）
                            predicted_spans.append(get_best_span_simple(start_logits, end_logits))  #得到预测位置   11， 14 ，对于单个，两个槽位可以得到同一个值 
                        else:
                            raise AssertionError()

                    target_spans = [[s,e] for (s,e) in zip(target["span_s"], target["span_e"])] #处理训练样本中的对应的真正角色的偏移值 
                    if len(target_spans)<len(predicted_spans):  #当前 角色样本中的真实角色实例的个数小于预测出来的值的个数（通过模板会预测出来一组值 ，可能个数比实际的多
                        # need to consider whether to make more 
                        pad_len = len(predicted_spans) - len(target_spans) #对于预测出来的跨度和目标跨度 数量 不一样的情况，按成本预测值 的数量 补齐，补为0值 
                        target_spans = target_spans + [[0,0]] * pad_len   # [ [s,e],[s,e]] 这样的格式 ，这是样本中的span
                        target["span_s"] = target["span_s"] + [0] * pad_len
                        target["span_e"] = target["span_e"] + [0] * pad_len
                        
                    if self.config.bipartite: # 两部匹配算法，也即匈牙利算法作进一步优化
                        idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)  #用两部匹配算法优化
                    else:  #不优化的，看下面
                        idx_preds = list(range(len(predicted_spans)))  #直接取顺序索引作为index.一对一
                        idx_targets = list(range(len(target_spans)))
                        if len(idx_targets) > len(idx_preds):
                            idx_targets = idx_targets[0:len(idx_preds)]  #对于 目标数大于预测数的情况（有可能？）直接对目标数的截前几个作为目标（跟预测数对齐）
                        idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                        idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                    cnt += len(idx_preds) #总共预测的槽位数（其实是模板中的槽位数，肯定一直有）
                    test=torch.stack(start_logits_list) #将列表拼成一个tensor, 原来的列表的两个元素变成新tensor的同一维的两向量
                    test2=test[idx_preds]  # 2*180 形状，两个向量的索引来源于indx_pred

                    tt=torch.LongTensor(target["span_s"]) #只取span_s这个列表转成向量
                    tt2=tt[idx_targets]  #通过indx_target去取，改index相当于换位置
                    # src= 2*180 和 target = 2 ,意思是，两维中，各对应一个值。 将180的argmax 找出来和  taget中对应维比较交叉熵得到损失。
                    start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds], torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets]) # start loss ，用交叉熵损失。 2*180  ， 2 ， 实际上是两行  180向量和对应的target值 比较。对180维向量取argmax得到对应的index,与target指定的true index 作交叉熵
                    end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds], torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])     # end loss 
                    batch_loss.append((start_loss + end_loss)/2)  #将个角色的loss ，分为start_loss/end_loss 加起来 除以2 ,得到本类角色的平均loss 作为本角色的loss
                
            logit_lists.append(output)
            if self.training: # inside batch mean loss
                total_loss = total_loss + torch.sum(torch.stack(batch_loss))/cnt   # cnt为每个样本中预测出来的所有角色的所有槽位数。 得到每个槽位的loss. 在预测时每个角色两个槽位，总共4种角色，总共就有2个槽位。总损失/ 8就是每个槽位的平均损失作为本样本的损失值 。
            
        if self.training:
            return total_loss/len(context_outputs), logit_lists   #输出的损失=总损失/当前批样本数。 logit_lists 就是要输出 的预测位置（argmax就是坐标啦）
        else:
            return [], logit_lists  #对于推理时，只需要直接输出 
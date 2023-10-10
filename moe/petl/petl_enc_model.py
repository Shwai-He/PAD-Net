from transformers import PreTrainedModel, RobertaConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)


class PETLEncModel(PreTrainedModel):
    def __init__(self, config, args, pretrained_model):
        super().__init__(config)
        self.args = args
        self.pretrained_model = pretrained_model

        if isinstance(config, RobertaConfig):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.decoder_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

    def check_params(self, module_name, safe_list, all_match=True):
        check = [partial_name in module_name for partial_name in safe_list]
        return all(check) if all_match else any(check)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):

        bsz = input_ids.shape[0]
        prefix_state = None
        output = self.pretrained_model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    labels=labels,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,
                                    prefix_state=prefix_state,
                                    )
        return output

    def reset_buffers(self):
        name_list, buf_list = [], []
        for name, module in self.named_modules():
            for name, buf in module.named_buffers():
                if 'mask' in name:
                    print(buf.mean())
                    name_list.append(name.split('.')[-1])
                    buf_list.append(buf)
            for i in range(len(name_list)):
                module.register_buffer(name_list[i], buf_list[i])


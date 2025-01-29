def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Get word embedding from self.word_embedding into input_embeds.
    inputs_embeds = self.word_embedding(input_ids)
    print("inputs_embeds")
 

    # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)
    print("pos_embeds")


    # Get token type ids. Since we are not considering token type, this embedding is
    # just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)
    
    # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    total_embeds = inputs_embeds + pos_embeds + tk_type_embeds
    layer_norm = self.embed_layer_norm(total_embeds)
    embed_output = self.embed_dropout(layer_norm)
    return embed_output
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from typing import Optional, Tuple


def reshape_state(state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reshape the hidden state of a bidirectional LSTM.
    
    Args:
        state: Tuple of hidden states, (h_state, c_state) each with shape 
               (num_layers * num_directions, batch_size, hidden_size)
               
    Returns:
        new_state: Tuple of reshaped hidden states, (new_h_state, new_c_state) each with shape
                   (num_layers, batch_size, 2*hidden_size)
    """
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
    ):
        """
        Luong et al. general attention (https://arxiv.org/pdf/1508.04025.pdf)
        Args:
            hidden_size: the size of hidden feature
        """
        super(Attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self, query:torch.Tensor, encoder_outputs:torch.Tensor, src_lengths:torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the attention output.
        
        Args:
            query: Tensor of shape (batch_size, 1, hidden_size) representing the current decoder state
            encoder_outputs: Tensor of shape (batch_size, max_src_len, hidden_size) representing the encoder outputs
            src_lengths: Tensor of shape (batch_size) representing the length of each input sequence
        Returns:
            attn_out: Tensor of shape (batch_size, 1, hidden_size) representing the attention output
        """
        query = query.squeeze(1)
        attn = torch.bmm(encoder_outputs, self.W(query).unsqueeze(-1)) # (batch_size, max_src_len, 1)
        attention_mask = ~self.sequence_mask(src_lengths)
        attention_mask = attention_mask.to(attn.device)
        attn.data.masked_fill_(attention_mask.unsqueeze(-1), -float('inf'))
        attention_weights = torch.softmax(attn, dim=1)
        context = torch.bmm(attention_weights, encoder_outputs).squeeze(1)
        attn_out = torch.tanh(self.linear_out(torch.cat((query, context), dim=-1))).unsqueeze(1)
        return attn_out

    def sequence_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Creates a boolean mask from sequence lengths.
        
        Args:
            lengths: Tensor of shape (batch_size) representing the length of each sequence
            
        Returns:
            mask: Tensor of shape (batch_size, max_len) containing the boolean mask, where False
                  indicates that a certain position in the sequence is padded.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (
            torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
        )


class Encoder(nn.Module):
    def __init__(self, src_vocab_size: int, hidden_size: int, 
        padding_idx: int, dropout: float) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, src:torch.Tensor, lengths:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass on the input.
        
        Args:
            src: Tensor of shape (batch_size, max_src_len) representing the input sequences
            lengths: Tensor of shape (batch_size) representing the length of each input sequence
            
        Returns:
            outputs: Tensor of shape (batch_size, max_src_len, hidden_size) representing the encoder outputs
            hidden: Tuple of two tensors, each of shape (num_layers * num_directions, batch_size, hidden_size)
                      representing the hidden state of the LSTM
        """
        embedded = self.dropout(self.embedding(src))
        packed = pack(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.lstm(packed)
        outputs, _ = unpack(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = self.dropout(outputs)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, tgt_vocab_size: int, attn: Attention, padding_idx: int, dropout: float):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(self, tgt: torch.Tensor, dec_state: Tuple[torch.Tensor, torch.Tensor], 
        encoder_outputs: torch.Tensor, src_lengths:torch.Tensor
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform a forward pass on the input.
        
        Args:
            tgt: Tensor of shape (batch_size, max_tgt_len) representing the target sequences
            dec_state: Tuple of two tensors, each of shape (num_layers * num_directions, batch_size, hidden_size)
                        representing the hidden state of the LSTM
            encoder_outputs: Tensor of shape (batch_size, max_src_len, hidden_size) representing the encoder outputs
            src_lengths: Tensor of shape (batch_size) representing the length of each input sequence
        
        Returns:
            outputs: Tensor of shape (batch_size* max_tgt_len, hidden_size) representing the decoder outputs after applying attention mechanism
            dec_state: Tuple of two tensors, each of shape (num_layers, batch_size, hidden_size) representing the hidden state of the LSTM
        """
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)
        tgt = tgt.unsqueeze(0) if len(tgt.shape) == 1 else tgt
        if tgt.shape[1] > 1:
            tgt = tgt[:,:-1]
        embedded = self.dropout(self.embedding(tgt))
        output, dec_state = self.lstm(embedded, dec_state)
        outputs = self.dropout(output.contiguous().view(-1, self.lstm.hidden_size))
        return outputs, dec_state


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        """
        Initialize the Seq2Seq model.
        
        Args:
            encoder: An instance of the Encoder class
            decoder: An instance of the Decoder class
        """
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(self, src: torch.Tensor, src_lengths:torch.Tensor, tgt: torch.Tensor, 
        dec_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform a forward pass on the input.
        
        Args:
            src: Tensor of shape (batch_size, max_src_len) representing the input sequences
            src_lengths: Tensor of shape (batch_size) representing the length of each input sequence
            tgt: Tensor of shape (batch_size, max_tgt_len) representing the target sequences
            dec_hidden: Tuple of two tensors, each of shape (num_layers, batch_size, hidden_size)
                         representing the hidden state of the LSTM
        
        Returns:
            output: Tensor of shape (batch_size* max_tgt_len, tgt_vocab_size) representing the generated sequence after applying attention mechanism and Linear Layer
            dec_hidden: Tuple of two tensors, each of shape (num_layers, batch_size, hidden_size) representing the hidden state of the LSTM
        """
        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden

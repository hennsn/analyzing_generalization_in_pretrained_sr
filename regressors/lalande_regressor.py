import sympy
import warnings
import numpy as np
import os
from collections import OrderedDict
import sklearn
import opt_consts
import torch
import numbers
import sys

class TransformerLalande(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):

    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        self.verbose = verbose
        self.random_state = random_state

        # https://github.com/omron-sinicx/transformer4sr
        sys.path.insert(0, 'transformer4sr')
        from transf_model.transformer_model import TransformerModel
        from transf_model._utils import is_tree_complete
        from transf_model._utils import translate_integers_into_tokens
        from datasets._utils import from_sympy_to_sequence, from_sequence_to_sympy, first_variables_first
        self.is_tree_complete = is_tree_complete
        self.from_sequence_to_sympy = from_sequence_to_sympy
        self.translate_integers_into_tokens = translate_integers_into_tokens
        self.from_sympy_to_sequence = from_sympy_to_sequence
        self.first_variables_first = first_variables_first

        self.transformer = TransformerModel(
            enc_type='mix',
            nb_samples=50,  # Number of samples par dataset
            max_nb_var=7,  # Max number of variables
            d_model=256,
            vocab_size=18+2,  # len(vocab) + padding token + <SOS> token
            seq_length=30,  # vocab_size + 1 - 1 (add <SOS> but shifted right)
            h=4,
            N_enc=4,
            N_dec=8,
            dropout=0.25,
        )
        del sys.path[0]
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        self.transformer = self.load_weights_()        
        self.X = None
        self.y = None
        #self.positives = []

    def load_weights_(self):
        # First reload big model
        PATH_WEIGHTS = os.path.join('transformer4sr', 'best_model_weights', 'mix_label_smoothing', 'model_weights.pt')
        hixon_state_dict = torch.load(PATH_WEIGHTS, map_location=torch.device('cpu'))

        my_state_dict = OrderedDict()
        for key in hixon_state_dict.keys():
            assert key[:7]=="module."
            my_state_dict[key[7:]] = hixon_state_dict[key]

        out = self.transformer.load_state_dict(my_state_dict, strict=True)
        self.transformer.eval()
        assert str(out) == '<All keys matched successfully>', 'Something went wrong while loading the transformer'
        return self.transformer

    def decode_with_transformer(self, transformer, dataset):
        """
        Greedy decode with the Transformer model.
        Decode until the equation tree is completed.
        Parameters:
        - transformer: torch Module object
        - dataset: tabular dataset
        shape = (batch_size=1, nb_samples=50, nb_max_var=7, 1)
        """
        encoder_output = transformer.encoder(dataset)  # Encoder output is fixed for the batch
        seq_length = transformer.decoder.positional_encoding.seq_length
        decoder_output = torch.zeros((dataset.shape[0], seq_length+1), dtype=torch.int64)  # initialize Decoder output
        decoder_output[:, 0] = 1
        is_complete = torch.zeros(dataset.shape[0], dtype=torch.bool)  # check when decoding is finished
        
        for n1 in range(seq_length):
            padding_mask = torch.eq(decoder_output[:, :-1], 0).unsqueeze(1).unsqueeze(1)
            future_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
            mask_dec = torch.logical_or(padding_mask, future_mask)
            temp = transformer.decoder(
                target_seq=decoder_output[:, :-1],
                mask_dec=mask_dec,
                output_enc=encoder_output,
            )
            temp = transformer.last_layer(temp)
            decoder_output[:, n1+1] = torch.where(is_complete, 0, torch.argmax(temp[:, n1], axis=-1))
            for n2 in range(dataset.shape[0]):
                if self.is_tree_complete(decoder_output[n2, 1:]):
                    is_complete[n2] = True
        return decoder_output
        
    def fit(self, X, y, verbose = 0, apply_formatting = True):
        # X can only be 6 dimensional at most! So we take first 6 dimensions
        
        y = y.flatten()
        assert len(y.shape) == 1
        
        self.y = y.copy()
        #self.positives = np.all(X > 0, axis = 0)

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        
        # turnoff warnings 
        warnings.filterwarnings("ignore")

        # Preprocessing 
        #data = np.column_stack([y, X[:, :6]])
        data = np.column_stack([X[:, :6], y])

        if apply_formatting:
            # original code taken from here: https://github.com/omron-sinicx/transformer4sr/blob/main/evaluate_model.py
            
            # filtering
            mask = np.all(data[:, :-1] > 0.0, axis=1)  # y is in the last column here
            valid_data = data[mask]

            # no filtering (effectively filters out inf values)
            #mask = np.all(data[:, :-1] != np.inf, axis=1)  # y is in the last column here
            #valid_data = data[mask]

            idx_rows = np.random.choice(valid_data.shape[0], 50, replace=True) #np.random.choice(valid_data.shape[0], 50, replace=False)
            shifts = np.zeros(valid_data.shape[1])
            new_dataset = np.zeros((50, 7))
            for k in range(valid_data.shape[1] - 1):  # y will be done separately at the end
                cur_data = valid_data[idx_rows, k]
                shifts[k+1] = np.mean(np.log10(cur_data))
                new_dataset[:, k+1] = np.power(10.0, np.log10(cur_data)-shifts[k+1])
            shifts[0] = np.mean(np.log10(np.abs(valid_data[idx_rows, -1])))  # maybe some negative values for y
            signs = np.where(valid_data[idx_rows, -1]<0.0, -1.0, 1.0)
            new_dataset[:, 0] = np.power(10.0, np.log10(np.abs(valid_data[idx_rows, -1])) - shifts[0]) * signs
        else:
            # option 1: take out random samples
            #idx_rows = np.random.choice(len(data), 50, replace=True) #np.random.choice(len(data), 50, replace=False)
            #new_dataset = data[idx_rows]
            
            # option 2: set to zeros
            #new_dataset = np.zeros((len(data), 7))
            #new_dataset[:, :data.shape[1]] = data

            # option 3: take out all samples
            new_dataset = data
        
        # Forward pass
        encoder_input = torch.Tensor(new_dataset).unsqueeze(0).unsqueeze(-1)
        decoder_output = self.decode_with_transformer(self.transformer, encoder_input)
        decoder_tokens = self.translate_integers_into_tokens(decoder_output[0])
        decoder_sympy = self.from_sequence_to_sympy(decoder_tokens)
        decoder_sympy = self.first_variables_first(decoder_sympy)
        decoder_sympy = sympy.simplify(sympy.factor(decoder_sympy))
        self.expr = decoder_sympy
        
        # Optimize constants
        # replace C -> c_0, xi -> x_i
        subs_dict = {}
        for s in self.expr.free_symbols:
            str_s = str(s)
            if str_s == 'C':
                subs_dict[s] = sympy.Symbol('c_0')
            else:
                idx = int(str_s[1:])
                if idx > X.shape[1]:
                    subs_dict[s] = 0
                else:
                    subs_dict[s] = sympy.Symbol(f'x_{idx-1}')
        
        self.expr = self.expr.subs(subs_dict)

        # optimize constants
        self.expr = opt_consts.fit_constants(self.expr, X, y)

        for x in self.expr.free_symbols:
            idx = int(str(x).split('_')[-1])
            
            #if self.positives[idx]:
            #    self.expr = self.expr.subs(x, sympy.Symbol(str(x), positive = True))
            
            self.expr = self.expr.subs(x, sympy.Symbol(str(x)))

        x_symbs = [f'x_{i}' for i in range(X.shape[1])]
        self.exec_func = sympy.lambdify(x_symbs, self.expr)

        return self
        
    def predict(self, X):
        assert hasattr(self, 'expr')

        if not hasattr(self, 'exec_func'):
            x_symbs = [f'x_{i}' for i in range(X.shape[1])]
            self.exec_func = sympy.lambdify(x_symbs, self.expr)

        pred = self.exec_func(*[X[:, i] for i in range(X.shape[1])])
        if isinstance(pred, numbers.Number):
            pred = pred*np.ones(X.shape[0])
        return pred

    def model(self):
        assert hasattr(self, 'expr')
        return self.expr
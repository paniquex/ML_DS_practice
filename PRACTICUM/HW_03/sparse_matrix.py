class CooSparseMatrix:
    def __init__(self, ijx_list, shape):
        self.ijx_list = ijx_list
        self.ijx_dict = {}
        self.shape = shape
        for ijx in self.ijx_list:
            ij, x = ijx[:2], ijx[2]
            is_i_int = isinstance(ij[0], int)
            is_j_int = isinstance(ij[1], int)
            if (not is_i_int) | (not is_j_int):
                raise TypeError
            if ij in self.ijx_dict.keys():
                raise TypeError
            else:
                self.ijx_dict[ij] = x

    def __setattr__(self, name, value):
        if name == 'shape':
            if 'shape' not in self.__dict__:
                self.__dict__[name] = value
            else:
                elems_amount = self.shape[0] * self.shape[1]
                if not isinstance(value, tuple):
                    raise TypeError
                elif not isinstance(value[0], int):
                    raise TypeError
                elif not isinstance(value[1], int):
                    raise TypeError
                elif (value[0] <= 0) | (value[1] <= 0):
                    raise TypeError
                elif (value[0] * value[1] != elems_amount):
                    raise TypeError
                else:
                    position_in_1d_2d = \
                        {i * self.shape[1] + j:
                            (i, j) for i, j in self.ijx_dict.keys()}
                    ijx_dict_new = {}
                    for val in position_in_1d_2d.keys():
                        new_i = int(val / value[1])
                        new_j = val - value[1] * new_i
                        ijx_dict_new[(new_i, new_j)] = \
                            self.ijx_dict[position_in_1d_2d[val]]
                    self.ijx_dict = ijx_dict_new
                    self.__dict__[name] = value
        elif name == 'T':
            raise AttributeError
        else:
            self.__dict__[name] = value

    def __getattr__(self, item):
        if item == 'T':
            ijx_list_new = [(ijx[1],
                             ijx[0],
                             self.ijx_dict[ijx]
                             ) for ijx in self.ijx_dict.keys()]
            shape_new = self.shape[1], self.shape[0]
            sparse_transposed = CooSparseMatrix(ijx_list_new, shape_new)
            return sparse_transposed
        else:
            return self.__dict__[item]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ijx_list_new = []
            if (idx >= self.shape[0]) | (idx < 0):
                raise TypeError
            for ij in self.ijx_dict.keys():
                if idx == ij[0]:
                    x = self.ijx_dict[ij]
                    ijx_list_new.append((0, ij[1], x))
            shape_new = (1, self.shape[1])
            sparse_new = CooSparseMatrix(ijx_list_new, shape_new)
            return sparse_new
        else:
            row_amount = self.shape[0]
            col_amount = self.shape[1]
            is_idx = (idx[0] in range(row_amount)) & \
                     (idx[1] in range(col_amount))
            if idx in self.ijx_dict.keys():
                return self.ijx_dict[idx]
            elif is_idx:
                return 0
            else:
                raise TypeError

    def __setitem__(self, ij, x):
        row_amount = self.shape[0]
        col_amount = self.shape[1]
        is_idx = (ij[0] in range(row_amount)) & \
                 (ij[1] in range(col_amount))
        if (x == 0) & (ij in self.ijx_dict.keys()):
            self.ijx_dict.pop(ij)
        if is_idx & (x != 0):
            self.ijx_dict[ij] = x
        elif not is_idx:
            raise KeyError(ij)

    def __add__(self, other):
        if self.shape != other.shape:
            raise TypeError
        if len(self.ijx_dict.keys()) == 0:
            return other
        if len(other.ijx_dict.keys()) == 0:
            return self
        ij_wout_zero = other.ijx_dict.keys() | self.ijx_dict.keys()
        sparse_new = CooSparseMatrix([], self.shape)
        for ij in ij_wout_zero:
            sparse_new[ij] = self[ij] + other[ij]
        return sparse_new

    def __sub__(self, other):
        if self.shape != other.shape:
            raise TypeError
        if self.ijx_dict == other.ijx_dict:
            return CooSparseMatrix([], self.shape)
        if len(other.ijx_dict.keys()) == 0:
            return self
        ij_wout_zero = other.ijx_dict.keys() | self.ijx_dict.keys()
        sparse_new = CooSparseMatrix([], self.shape)
        for ij in ij_wout_zero:
            if self[ij] != other[ij]:
                sparse_new[ij] = self[ij] - other[ij]
        return sparse_new

    def __mul__(self, other):
        sparse_new = CooSparseMatrix([], self.shape)
        if other == 0:
            return sparse_new
        for ij in self.ijx_dict.keys():
            sparse_new[ij] = self[ij] * other
        return sparse_new

    def __rmul__(self, other):
        sparse_new = CooSparseMatrix([], self.shape)
        if other == 0:
            return sparse_new
        for ij in self.ijx_dict.keys():
            sparse_new[ij] = self[ij] * other
        return sparse_new

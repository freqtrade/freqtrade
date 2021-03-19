import ast


class HyperParser(ast.NodeTransformer):
    def __init__(self, obj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj = obj

    @property
    def class_def(self):
        return [cl.body for cl in self.obj.body if isinstance(cl, ast.ClassDef)][0]

    @property
    def indicators(self):
        return [ind for ind in self.class_def]
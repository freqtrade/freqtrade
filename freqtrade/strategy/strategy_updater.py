import ast
import os
import shutil


class strategy_updater:
    name_mapping = {
        'ticker_interval': 'timeframe',
        'buy': 'enter_long',
        'sell': 'exit_long',
        'buy_tag': 'enter_tag',
        'sell_reason': 'exit_reason',

        'sell_signal': 'exit_signal',
        'custom_sell': 'custom_exit',
        'force_sell': 'force_exit',
        'emergency_sell': 'emergency_exit',

        # Strategy/config settings:
        'use_sell_signal': 'use_exit_signal',
        'sell_profit_only': 'exit_profit_only',
        'sell_profit_offset': 'exit_profit_offset',
        'ignore_roi_if_buy_signal': 'ignore_roi_if_entry_signal',
        'forcebuy_enable': 'force_entry_enable',
    }

    function_mapping = {
        'populate_buy_trend': 'populate_entry_trend',
        'populate_sell_trend': 'populate_exit_trend',
        'custom_sell': 'custom_exit',
        'check_buy_timeout': 'check_entry_timeout',
        'check_sell_timeout': 'check_exit_timeout',
        # '': '',
    }
    # order_time_in_force, order_types, unfilledtimeout
    otif_ot_unfilledtimeout = {
        'buy': 'entry',
        'sell': 'exit',
    }

    # create a dictionary that maps the old column names to the new ones
    rename_dict = {'buy': 'enter_long', 'sell': 'exit_long', 'buy_tag': 'enter_tag'}

    def start(self, strategy_obj: dict) -> None:
        """
        Run strategy updater
        It updates a strategy to v3 with the help of the ast-module
        :return: None
        """

        self.cwd = os.getcwd()
        self.strategies_backup_folder = f'{os.getcwd()}user_data/strategies_orig_updater'
        source_file = strategy_obj['location']

        # read the file
        with open(source_file, 'r') as f:
            old_code = f.read()
        if not os.path.exists(self.strategies_backup_folder):
            os.makedirs(self.strategies_backup_folder)

        # backup original
        # => currently no date after the filename,
        # could get overridden pretty fast if this is fired twice!
        shutil.copy(source_file, f"{self.strategies_backup_folder}/{strategy_obj['location_rel']}")

        # update the code
        new_code = strategy_updater.update_code(self, old_code,
                                                strategy_updater.name_mapping,
                                                strategy_updater.function_mapping,
                                                strategy_updater.rename_dict)

        # write the modified code to the destination folder
        with open(source_file, 'w') as f:
            f.write(new_code)
        print(f"conversion of file {source_file} successful.")

    # define the function to update the code
    def update_code(self, code, _name_mapping, _function_mapping, _rename_dict):
        # parse the code into an AST
        tree = ast.parse(code)

        # use the AST to update the code
        updated_code = strategy_updater.modify_ast(
            tree,
            _name_mapping,
            _function_mapping,
            _rename_dict)

        # return the modified code without executing it
        return updated_code

    # function that uses the ast module to update the code
    def modify_ast(node, _name_mapping, _function_mapping, _rename_dict): # noqa
        # create a visitor that will update the names and functions
        class NameUpdater(ast.NodeTransformer):
            def generic_visit(self, node):
                # traverse the AST recursively by calling the visitor method for each child node
                if hasattr(node, "_fields"):
                    for field_name, field_value in ast.iter_fields(node):
                        self.check_fields(field_value)
                        self.check_strategy_and_config_settings(field_value)
                        # add this check to handle the case where field_value is a slice
                        if isinstance(field_value, ast.Slice):
                            self.visit(field_value)
                        # add this check to handle the case where field_value is a target
                        if isinstance(field_value, ast.expr_context):
                            self.visit(field_value)

            def check_fields(self, field_value):
                if isinstance(field_value, list):
                    for item in field_value:
                        if isinstance(item, ast.AST):
                            self.visit(item)

            def check_strategy_and_config_settings(self, field_value):
                if (isinstance(field_value, ast.AST) and
                        hasattr(node, "targets") and
                        isinstance(node.targets, list)):
                    for target in node.targets:
                        if (hasattr(target, "id") and
                                (target.id == "order_time_in_force" or
                                 target.id == "order_types" or
                                 target.id == "unfilledtimeout") and
                                hasattr(field_value, "keys") and
                                isinstance(field_value.keys, list)):
                            for key in field_value.keys:
                                self.visit(key)

            def visit_Name(self, node):
                # if the name is in the mapping, update it
                if node.id in _name_mapping:
                    node.id = _name_mapping[node.id]
                return node

            def visit_Import(self, node):
                # do not update the names in import statements
                return node

            def visit_ImportFrom(self, node):
                # do not update the names in import statements
                if hasattr(node, "module"):
                    if node.module == "freqtrade.strategy.hyper":
                        node.module = "freqtrade.strategy"
                return node

            def visit_FunctionDef(self, node):
                # if the function name is in the mapping, update it
                if node.name in _function_mapping:
                    node.name = _function_mapping[node.name]
                return self.generic_visit(node)

            def visit_Attribute(self, node):
                # if the attribute name is 'nr_of_successful_buys',
                # update it to 'nr_of_successful_entries'
                if isinstance(node.value, ast.Name) and \
                        node.value.id == 'trades' and \
                        node.attr == 'nr_of_successful_buys':
                    node.attr = 'nr_of_successful_entries'
                return self.generic_visit(node)

            def visit_ClassDef(self, node):
                # check if the class is derived from IStrategy
                if any(isinstance(base, ast.Name) and
                       base.id == 'IStrategy' for base in node.bases):
                    # check if the INTERFACE_VERSION variable exists
                    has_interface_version = any(
                        isinstance(child, ast.Assign) and
                        isinstance(child.targets[0], ast.Name) and
                        child.targets[0].id == 'INTERFACE_VERSION'
                        for child in node.body
                    )

                    # if the INTERFACE_VERSION variable does not exist, add it as the first child
                    if not has_interface_version:
                        node.body.insert(0, ast.parse('INTERFACE_VERSION = 3').body[0])
                    # otherwise, update its value to 3
                    else:
                        for child in node.body:
                            if isinstance(child, ast.Assign) and \
                                    isinstance(child.targets[0], ast.Name) and \
                                    child.targets[0].id == 'INTERFACE_VERSION':
                                child.value = ast.parse('3').body[0].value
                return self.generic_visit(node)

            def visit_Subscript(self, node):
                if isinstance(node.slice, ast.Constant):
                    if node.slice.value in strategy_updater.rename_dict:
                        # Replace the slice attributes with the values from rename_dict
                        node.slice.value = strategy_updater.rename_dict[node.slice.value]
                if hasattr(node.slice, "elts"):
                    for elt in node.slice.elts:
                        if isinstance(elt, ast.Constant) and \
                                elt.value in strategy_updater.rename_dict:
                            elt.value = strategy_updater.rename_dict[elt.value]
                return node

            def visit_Constant(self, node):
                # do not update the names in import statements
                if node.value in \
                        strategy_updater.otif_ot_unfilledtimeout:
                    node.value = \
                        strategy_updater.otif_ot_unfilledtimeout[node.value]
                return node

        # use the visitor to update the names and functions in the AST
        NameUpdater().visit(node)

        # first fix the comments so it understands "\n" properly inside multi line comments.
        ast.fix_missing_locations(node)
        ast.increment_lineno(node, n=1)

        # generate the new code from the updated AST
        return ast.unparse(node)

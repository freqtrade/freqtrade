import ast
import os
import shutil
from pathlib import Path

import astor


class StrategyUpdater:
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

    def start(self, config, strategy_obj: dict) -> None:
        """
        Run strategy updater
        It updates a strategy to v3 with the help of the ast-module
        :return: None
        """

        source_file = strategy_obj['location']
        print(f"started conversion of {source_file}")
        strategies_backup_folder = Path.joinpath(config['user_data_dir'], "strategies_orig_updater")
        target_file = Path.joinpath(strategies_backup_folder, strategy_obj['location_rel'])

        # read the file
        with open(source_file, 'r') as f:
            old_code = f.read()
        if not os.path.exists(strategies_backup_folder):
            os.makedirs(strategies_backup_folder)

        # backup original
        # => currently no date after the filename,
        # could get overridden pretty fast if this is fired twice!
        # The folder is always the same and the file name too (currently).
        shutil.copy(source_file, target_file)

        # update the code
        new_code = StrategyUpdater.update_code(self, old_code)
        # write the modified code to the destination folder
        with open(source_file, 'w') as f:
            f.write(new_code)
        print(f"conversion of file {source_file} successful.")

    # define the function to update the code
    def update_code(self, code):
        # parse the code into an AST
        tree = ast.parse(code)

        # use the AST to update the code
        updated_code = self.modify_ast(tree)

        # return the modified code without executing it
        return updated_code

    # function that uses the ast module to update the code
    def modify_ast(self, tree):  # noqa
        # use the visitor to update the names and functions in the AST
        NameUpdater().visit(tree)

        # first fix the comments, so it understands "\n" properly inside multi line comments.
        ast.fix_missing_locations(tree)
        ast.increment_lineno(tree, n=1)

        # generate the new code from the updated AST
        return astor.to_source(tree)


# Here we go through each respective node, slice, elt, key ... to replace outdated entries.
class NameUpdater(ast.NodeTransformer):
    def generic_visit(self, node):
        # traverse the AST recursively by calling the visitor method for each child node
        if hasattr(node, "_fields"):
            for field_name, field_value in ast.iter_fields(node):
                self.check_strategy_and_config_settings(node, field_value)
                self.check_fields(field_value)
                for child in ast.iter_child_nodes(node):
                    self.generic_visit(child)

    def check_fields(self, field_value):
        if isinstance(field_value, list):
            for item in field_value:
                if isinstance(item, ast.AST) or isinstance(item, ast.If):
                    self.visit(item)
        if isinstance(field_value, ast.Name):
            self.visit_Name(field_value)

    def check_strategy_and_config_settings(self, node, field_value):
        if (isinstance(field_value, ast.AST) and
                hasattr(node, "targets") and
                isinstance(node.targets, list)):
            for target in node.targets:
                if (hasattr(target, "id") and
                        hasattr(field_value, "keys") and
                        isinstance(field_value.keys, list)):
                    if (target.id == "order_time_in_force" or
                            target.id == "order_types" or
                            target.id == "unfilledtimeout"):
                        for key in field_value.keys:
                            self.visit(key)

    def check_args(self, node):
        if isinstance(node.args, ast.arguments):
            self.check_args(node.args)
        if hasattr(node, "args"):
            if isinstance(node.args, list):
                for arg in node.args:
                    arg.arg = StrategyUpdater.name_mapping[arg.arg]
        return node

    def visit_Name(self, node):
        # if the name is in the mapping, update it
        if node.id in StrategyUpdater.name_mapping:
            node.id = StrategyUpdater.name_mapping[node.id]
        return node

    def visit_Import(self, node):
        # do not update the names in import statements
        return node

    # This function is currently never successfully triggered
    # since freqtrade currently only allows valid code to be processed.
    # The module .hyper does not anymore exist and by that fails to even
    # reach this function to be updated currently.
    def visit_ImportFrom(self, node):
        # if hasattr(node, "module"):
        #    if node.module == "freqtrade.strategy.hyper":
        #        node.module = "freqtrade.strategy"
        return node

    def visit_If(self, node: ast.If):
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # if the function name is in the mapping, update it
        if node.name in StrategyUpdater.function_mapping:
            node.name = StrategyUpdater.function_mapping[node.name]
        if hasattr(node, "args"):
            self.check_args(node)
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if hasattr(node, "targets") and isinstance(node.targets, list):
            for target in node.targets:
                if hasattr(target, "id") and target.id in StrategyUpdater.name_mapping:
                    target.id = StrategyUpdater.name_mapping[target.id]
        return node

    def visit_Attribute(self, node):
        # if the attribute name is 'nr_of_successful_buys',
        # update it to 'nr_of_successful_entries'
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == 'trades'
            and node.attr == 'nr_of_successful_buys'
        ):
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
                    if (
                        isinstance(child, ast.Assign)
                        and isinstance(child.targets[0], ast.Name)
                        and child.targets[0].id == 'INTERFACE_VERSION'
                    ):
                        child.value = ast.parse('3').body[0].value
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Constant):
            if node.slice.value in StrategyUpdater.rename_dict:
                # Replace the slice attributes with the values from rename_dict
                node.slice.value = StrategyUpdater.rename_dict[node.slice.value]
        if hasattr(node.slice, "elts"):
            self.visit_elts(node.slice.elts)
        if hasattr(node.slice, "value"):
            if hasattr(node.slice.value, "elts"):
                self.visit_elts(node.slice.value.elts)
        # Check if the target is a Subscript object with a "value" attribute
        # if isinstance(target, ast.Subscript) and hasattr(target.value, "attr"):
        #    if target.value.attr == "loc":
        #        self.visit(target)
        return node

    # elts can have elts (technically recursively)
    def visit_elts(self, elts):
        if isinstance(elts, list):
            for elt in elts:
                self.visit_elt(elt)
        else:
            self.visit_elt(elts)

    # sub function again needed since the structure itself is highly flexible ...
    def visit_elt(self, elt):
        if isinstance(elt, ast.Constant) and elt.value in StrategyUpdater.rename_dict:
            elt.value = StrategyUpdater.rename_dict[elt.value]
        if hasattr(elt, "elts"):
            self.visit_elts(elt.elts)
        if hasattr(elt, "args"):
            if isinstance(elt.args, ast.arguments):
                self.visit_elts(elt.args)
            else:
                for arg in elt.args:
                    self.visit_elts(arg)

    def visit_Constant(self, node):
        # do not update the names in import statements
        if node.value in StrategyUpdater.otif_ot_unfilledtimeout:
            node.value = StrategyUpdater.otif_ot_unfilledtimeout[node.value]
        return node

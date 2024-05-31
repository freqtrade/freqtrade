import shutil
from pathlib import Path

import ast_comments

from freqtrade.constants import Config


class StrategyUpdater:
    name_mapping = {
        "ticker_interval": "timeframe",
        "buy": "enter_long",
        "sell": "exit_long",
        "buy_tag": "enter_tag",
        "sell_reason": "exit_reason",
        "sell_signal": "exit_signal",
        "custom_sell": "custom_exit",
        "force_sell": "force_exit",
        "emergency_sell": "emergency_exit",
        # Strategy/config settings:
        "use_sell_signal": "use_exit_signal",
        "sell_profit_only": "exit_profit_only",
        "sell_profit_offset": "exit_profit_offset",
        "ignore_roi_if_buy_signal": "ignore_roi_if_entry_signal",
        "forcebuy_enable": "force_entry_enable",
    }

    function_mapping = {
        "populate_buy_trend": "populate_entry_trend",
        "populate_sell_trend": "populate_exit_trend",
        "custom_sell": "custom_exit",
        "check_buy_timeout": "check_entry_timeout",
        "check_sell_timeout": "check_exit_timeout",
        # '': '',
    }
    # order_time_in_force, order_types, unfilledtimeout
    otif_ot_unfilledtimeout = {
        "buy": "entry",
        "sell": "exit",
    }

    # create a dictionary that maps the old column names to the new ones
    rename_dict = {"buy": "enter_long", "sell": "exit_long", "buy_tag": "enter_tag"}

    def start(self, config: Config, strategy_obj: dict) -> None:
        """
        Run strategy updater
        It updates a strategy to v3 with the help of the ast-module
        :return: None
        """

        source_file = strategy_obj["location"]
        strategies_backup_folder = Path.joinpath(config["user_data_dir"], "strategies_orig_updater")
        target_file = Path.joinpath(strategies_backup_folder, strategy_obj["location_rel"])

        # read the file
        with Path(source_file).open("r") as f:
            old_code = f.read()
        if not strategies_backup_folder.is_dir():
            Path(strategies_backup_folder).mkdir(parents=True, exist_ok=True)

        # backup original
        # => currently no date after the filename,
        # could get overridden pretty fast if this is fired twice!
        # The folder is always the same and the file name too (currently).
        shutil.copy(source_file, target_file)

        # update the code
        new_code = self.update_code(old_code)
        # write the modified code to the destination folder
        with Path(source_file).open("w") as f:
            f.write(new_code)

    # define the function to update the code
    def update_code(self, code):
        # parse the code into an AST
        tree = ast_comments.parse(code)

        # use the AST to update the code
        updated_code = self.modify_ast(tree)

        # return the modified code without executing it
        return updated_code

    # function that uses the ast module to update the code
    def modify_ast(self, tree):  # noqa
        # use the visitor to update the names and functions in the AST
        NameUpdater().visit(tree)

        # first fix the comments, so it understands "\n" properly inside multi line comments.
        ast_comments.fix_missing_locations(tree)
        ast_comments.increment_lineno(tree, n=1)

        # generate the new code from the updated AST
        # without indent {} parameters would just be written straight one after the other.

        # ast_comments would be amazing since this is the only solution that carries over comments,
        # but it does currently not have an unparse function, hopefully in the future ... !
        # return ast_comments.unparse(tree)

        return ast_comments.unparse(tree)


# Here we go through each respective node, slice, elt, key ... to replace outdated entries.
class NameUpdater(ast_comments.NodeTransformer):
    def generic_visit(self, node):
        # space is not yet transferred from buy/sell to entry/exit and thereby has to be skipped.
        if isinstance(node, ast_comments.keyword):
            if node.arg == "space":
                return node

        # from here on this is the original function.
        for field, old_value in ast_comments.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast_comments.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast_comments.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast_comments.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit_Expr(self, node):
        if hasattr(node.value, "left") and hasattr(node.value.left, "id"):
            node.value.left.id = self.check_dict(StrategyUpdater.name_mapping, node.value.left.id)
            self.visit(node.value)
        return node

    # Renames an element if contained inside a dictionary.
    @staticmethod
    def check_dict(current_dict: dict, element: str):
        if element in current_dict:
            element = current_dict[element]
        return element

    def visit_arguments(self, node):
        if isinstance(node.args, list):
            for arg in node.args:
                arg.arg = self.check_dict(StrategyUpdater.name_mapping, arg.arg)
        return node

    def visit_Name(self, node):
        # if the name is in the mapping, update it
        node.id = self.check_dict(StrategyUpdater.name_mapping, node.id)
        return node

    def visit_Import(self, node):
        # do not update the names in import statements
        return node

    def visit_ImportFrom(self, node):
        # if hasattr(node, "module"):
        #    if node.module == "freqtrade.strategy.hyper":
        #        node.module = "freqtrade.strategy"
        return node

    def visit_If(self, node: ast_comments.If):
        for child in ast_comments.iter_child_nodes(node):
            self.visit(child)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.check_dict(StrategyUpdater.function_mapping, node.name)
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        if (
            isinstance(node.value, ast_comments.Name)
            and node.value.id == "trade"
            and node.attr == "nr_of_successful_buys"
        ):
            node.attr = "nr_of_successful_entries"
        return node

    def visit_ClassDef(self, node):
        # check if the class is derived from IStrategy
        if any(
            isinstance(base, ast_comments.Name) and base.id == "IStrategy" for base in node.bases
        ):
            # check if the INTERFACE_VERSION variable exists
            has_interface_version = any(
                isinstance(child, ast_comments.Assign)
                and isinstance(child.targets[0], ast_comments.Name)
                and child.targets[0].id == "INTERFACE_VERSION"
                for child in node.body
            )

            # if the INTERFACE_VERSION variable does not exist, add it as the first child
            if not has_interface_version:
                node.body.insert(0, ast_comments.parse("INTERFACE_VERSION = 3").body[0])
            # otherwise, update its value to 3
            else:
                for child in node.body:
                    if (
                        isinstance(child, ast_comments.Assign)
                        and isinstance(child.targets[0], ast_comments.Name)
                        and child.targets[0].id == "INTERFACE_VERSION"
                    ):
                        child.value = ast_comments.parse("3").body[0].value
        self.generic_visit(node)
        return node

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast_comments.Constant):
            if node.slice.value in StrategyUpdater.rename_dict:
                # Replace the slice attributes with the values from rename_dict
                node.slice.value = StrategyUpdater.rename_dict[node.slice.value]
        if hasattr(node.slice, "elts"):
            self.visit_elts(node.slice.elts)
        if hasattr(node.slice, "value"):
            if hasattr(node.slice.value, "elts"):
                self.visit_elts(node.slice.value.elts)
        return node

    # elts can have elts (technically recursively)
    def visit_elts(self, elts):
        if isinstance(elts, list):
            for elt in elts:
                self.visit_elt(elt)
        else:
            self.visit_elt(elts)
        return elts

    # sub function again needed since the structure itself is highly flexible ...
    def visit_elt(self, elt):
        if isinstance(elt, ast_comments.Constant) and elt.value in StrategyUpdater.rename_dict:
            elt.value = StrategyUpdater.rename_dict[elt.value]
        if hasattr(elt, "elts"):
            self.visit_elts(elt.elts)
        if hasattr(elt, "args"):
            if isinstance(elt.args, ast_comments.arguments):
                self.visit_elts(elt.args)
            else:
                for arg in elt.args:
                    self.visit_elts(arg)
        return elt

    def visit_Constant(self, node):
        node.value = self.check_dict(StrategyUpdater.otif_ot_unfilledtimeout, node.value)
        node.value = self.check_dict(StrategyUpdater.name_mapping, node.value)
        return node

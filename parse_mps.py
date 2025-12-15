import re
import sys
import time

import cvxpy as cp
import numpy as np
import pulp
import torch


def parse_mps(lp_file_path):
    problem = pulp.LpProblem.fromMPS(lp_file_path, sense=pulp.LpMinimize)
    lps_problem = problem[1]

    # from what I can tell, this is slow
    return problem


def parse_self_MPS(path: str):
    """
    adapted from Julian Märte (https://github.com/pchtsp/pysmps)
    returns a dictionary with the contents of the model.
    This dictionary can be used to generate an LpProblem

    :param path: path of mps file
    :param sense: 1 for minimize, -1 for maximize
    :param dropConsNames: if True, do not store the names of constraints
    :return: a dictionary with all the problem data
    """
    CORE_FILE_ROW_MODE = "ROWS"
    CORE_FILE_COL_MODE = "COLUMNS"
    CORE_FILE_RHS_MODE = "RHS"
    CORE_FILE_BOUNDS_MODE = "BOUNDS"

    CORE_FILE_BOUNDS_MODE_NAME_GIVEN = "BOUNDS_NAME"
    CORE_FILE_BOUNDS_MODE_NO_NAME = "BOUNDS_NO_NAME"
    CORE_FILE_RHS_MODE_NAME_GIVEN = "RHS_NAME"
    CORE_FILE_RHS_MODE_NO_NAME = "RHS_NO_NAME"

    ROW_MODE_OBJ = "N"

    mode = ""
    problem_name = ""
    num_cosntraint_rows = 0
    num_vars = 0
    num_slack_vars = 0
    objecive_name = ""
    objective_coefficients = []
    var_name_to_col_index = {}
    var_name_to_constraint_index = {}
    constraint_name_to_row_index = {}
    constraint_matrix = []  # placeholder, will resize later
    right_hand_side = []
    rows_to_add_slack = []
    var_name_to_bounds = {}

    # parameters = MPSParameters(name="", sense=sense, status=0, sol_status=0)
    # variable_info: dict[str, MPSVariable] = {}
    # constraints: dict[str, MPSConstraint] = {}
    # objective = MPSObjective(name="", coefficients=[])
    # sos1: list[Any] = []
    # sos2: list[Any] = []
    # TODO: maybe take out rhs_names and bnd_names? not sure if they're useful
    # its techincally possible to have multiple RHS sets in MPS files, but no one does that
    rhs_names: list[str] = []
    bnd_names: list[str] = []
    # integral_marker: bool = False

    with open(path) as reader:
        for line in reader:
            line = re.split(" |\t", line)  # type: ignore[assignment]
            line = [x.strip() for x in line]  # type: ignore[assignment]
            line = list(filter(None, line))  # type: ignore[assignment]
            # print(line)

            if line[0] == "ENDATA":  # EOF
                break
            if line[0] == "*":  # comment line
                continue
            if line[0] == "NAME":  # problem name
                if len(line) > 1:
                    problem_name = line[1]
                else:
                    problem_name = ""
                continue
            #
            #         # here we get the mode
            if line[0] in [CORE_FILE_ROW_MODE, CORE_FILE_COL_MODE]:
                mode = line[0]
            elif line[0] == CORE_FILE_RHS_MODE and len(line) <= 2:
                right_hand_side = [0.0] * num_cosntraint_rows
                if len(line) > 1:
                    rhs_names.append(line[1])
                    mode = CORE_FILE_RHS_MODE_NAME_GIVEN
                else:
                    mode = CORE_FILE_RHS_MODE_NO_NAME
            elif line[0] == CORE_FILE_BOUNDS_MODE and len(line) <= 2:
                if len(line) > 1:
                    bnd_names.append(line[1])
                    mode = CORE_FILE_BOUNDS_MODE_NAME_GIVEN
                else:
                    mode = CORE_FILE_BOUNDS_MODE_NO_NAME

            # here we query the mode variable
            elif mode == CORE_FILE_ROW_MODE:
                row_type = line[0]
                row_name = line[1]
                if row_type == ROW_MODE_OBJ:
                    objective_name = row_name
                else:
                    if row_type in ["L", "G"]:
                        num_slack_vars += 1
                        rows_to_add_slack.append((num_cosntraint_rows, row_type))
                    constraint_name_to_row_index[row_name] = num_cosntraint_rows
                    constraint_matrix.append([])
                    num_cosntraint_rows += 1
            elif mode == CORE_FILE_COL_MODE:
                var_name = line[0]
                # if len(line) > 1 and line[1] == "'MARKER'": TODO: bro i have no clue what this is
                #     if line[2] == "'INTORG'":
                #         integral_marker = True
                #     elif line[2] == "'INTEND'":
                #         integral_marker = False
                #     continue
                if (
                    var_name not in var_name_to_col_index
                ):  # first time we see this variable
                    var_name_to_col_index[var_name] = len(var_name_to_col_index)
                    num_vars += 1
                    # variable_info[var_name] = MPSVariable(
                    #     cat=COL_EQUIV[integral_marker], name=var_name
                    # )
                j = 1
                while j < len(line) - 1:
                    if line[j] == objective_name:
                        # we store the variable objective coefficient
                        objective_coefficients.append(float(line[j + 1]))
                        # assert (
                        #     len(objective_coefficients) - 1
                        #     == var_name_to_col_index[var_name]
                        # )  # make sure the coefficients are in the right order
                        # objective.coefficients.append(
                        #     MPSCoefficient(name=var_name, value=float(line[j + 1]))
                        # )
                    else:
                        # we store the variable coefficient
                        var_name_to_col_index[var_name] = (
                            len(var_name_to_col_index)
                            if var_name not in var_name_to_col_index
                            else var_name_to_col_index[var_name]
                        )

                        row_index = constraint_name_to_row_index[line[j]]
                        if var_name not in var_name_to_constraint_index:
                            var_name_to_constraint_index[var_name] = []
                        var_name_to_constraint_index[var_name].append(
                            (row_index, float(line[j + 1]))
                        )
                        # assert (
                        #     len(constraint_matrix[row_index]) - 1
                        #     == var_name_to_col_index[var_name]
                        # )  # make sure the coefficients are in the right order
                        # constraints[line[j]].coefficients.append(
                        #     MPSCoefficient(name=var_name, value=float(line[j + 1]))
                        # )
                    j = j + 2
            # elif mode == CORE_FILE_RHS_MODE_NAME_GIVEN:
            # if line[0] != rhs_names[-1]:
            #     raise const.PulpError(
            #         "Other RHS name was given even though name was set after RHS tag."
            #     )
            # readMPSSetRhs(line, constraints)  # type: ignore[arg-type]
            elif mode == CORE_FILE_RHS_MODE_NO_NAME:

                # readMPSSetRhs(line, constraints)  # type: ignore[arg-type]
                if line[0] not in rhs_names:
                    rhs_names.append(line[0])
                j = 1
                while j < len(line) - 1:
                    row_name = line[j]
                    value = float(line[j + 1])
                    row_index = constraint_name_to_row_index[row_name]
                    right_hand_side[row_index] = value
                    j = j + 2
            # elif mode == CORE_FILE_BOUNDS_MODE_NAME_GIVEN:
            #     if line[1] != bnd_names[-1]:
            #         raise const.PulpError(
            #             "Other BOUNDS name was given even though name was set after BOUNDS tag."
            #         )
            #     readMPSSetBounds(line, variable_info)  # type: ignore[arg-type]
            elif mode == CORE_FILE_BOUNDS_MODE_NO_NAME:
                # readMPSSetBounds(line, variable_info)  # type: ignore[arg-type]
                if line[1] not in bnd_names:
                    bnd_names.append(line[1])
                j = 2
                while j < len(line) - 1:
                    var_name = line[j]
                    bound_type = line[0]
                    var_name_to_bounds[var_name] = (bound_type, float(line[j + 1]))
                    j = j + 2

    # INFO: construct the constraint matrix DENSE
    # constraint_matrix = torch.zeros(num_cosntraint_rows, num_vars + num_slack_vars)
    # for var_name, entries in var_name_to_constraint_index.items():
    #     col_index = var_name_to_col_index[var_name]
    #     for row_index, value in entries:
    #         constraint_matrix[row_index, col_index] = value
    # slack_var_index = num_vars
    # for row_index, row_type in rows_to_add_slack:
    #     if row_type == "L":
    #         constraint_matrix[row_index, slack_var_index] = 1.0
    #     elif row_type == "G":
    #         constraint_matrix[row_index, slack_var_index] = -1.0
    #     slack_var_index += 1
    # INFO: construct the constraint matrix SPARSE
    constraint_matrix = []
    # print("var_name_to_constraint_index:", var_name_to_constraint_index)
    for var_name, constraints in var_name_to_constraint_index.items():
        for entry in constraints:
            row_index, value = entry
            col_index = var_name_to_col_index[var_name]
            constraint_matrix.append((row_index, col_index, value))
    slack_var_index = num_vars
    for row_index, row_type in rows_to_add_slack:
        col_index = slack_var_index
        if row_type == "L":
            constraint_matrix.append((row_index, col_index, 1.0))
        elif row_type == "G":
            constraint_matrix.append((row_index, col_index, -1.0))
        slack_var_index += 1

    # print("Constraint Matrix:\n", constraint_matrix)

    # print("Problem Name:", problem_name)
    # print("Objective: ", objective_coefficients)
    # print("Constraints:", var_name_to_constraint_index)
    # print("RHS:", right_hand_side)
    # print("Bounds:", var_name_to_bound_type, bounds)

    # constraints_list = list(constraints.values())
    # if dropConsNames:
    # for c in constraints_list:
    # c.name = None
    # objective.name = None
    # variable_info_list = list(variable_info.values())
    return objective_coefficients, constraint_matrix, right_hand_side, var_name_to_bounds


def normalize_problem():
    pass


if __name__ == "__main__":
    lp_file_path = sys.argv[1]
    start_time = time.time()
    # problem = parse_mps(lp_file_path)
    problem = parse_self_MPS(lp_file_path)
    end_time = time.time()
    print(f"Time taken to parse MPS file: {end_time - start_time} seconds")
    print(problem)

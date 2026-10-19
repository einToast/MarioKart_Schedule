import random

def fill_round(round_fields, teams, num_fields, teams_per_game, team_appearances, field_1_appearances):
    round_teams = teams[:]
    random.shuffle(round_teams)
    for fld in range(num_fields):
        game_teams = round_teams[
            fld * teams_per_game : fld * teams_per_game + teams_per_game
        ]
        for team in game_teams:
            team_appearances[team] += 1
            if fld == 0:
                field_1_appearances[team] += 1
        round_fields[f"Field {chr(65 + fld)}"] = game_teams


def add_to_field_a(schedule, team, teams_per_game):
    for round_fields in schedule.values():
        if len(round_fields["Field A"]) < teams_per_game:
            round_fields["Field A"].append(team)
            return True
    return False


def generate_schedule(num_teams=25, num_fields=4, num_rounds=8, teams_per_game=4):
    teams = list(range(1, num_teams + 1))
    schedule = {
        f"Round {i + 1}": {f"Field {chr(65 + j)}": [] for j in range(num_fields)}
        for i in range(num_rounds)
    }
    team_appearances = dict.fromkeys(teams, 0)
    field_1_appearances = dict.fromkeys(teams, 0)

    for round_fields in schedule.values():
        fill_round(round_fields, teams, num_fields, teams_per_game, team_appearances, field_1_appearances)

    # Ensure each team plays at least once on Field 1
    for team in teams:
        if field_1_appearances[team] == 0 and add_to_field_a(schedule, team, teams_per_game):
            field_1_appearances[team] += 1

    return schedule

if __name__ == "__main__":
    # Generate and print the schedule
    for round_name, fields in generate_schedule().items():
        print(round_name)
        for field, field_teams in fields.items():
            print(f"  {field}: {field_teams}")

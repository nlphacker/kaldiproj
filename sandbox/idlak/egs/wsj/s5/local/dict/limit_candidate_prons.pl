#!/usr/bin/perl

# This program enforces the rule that
# if a "more specific" rule applies, we cannot use the more general rule.
# It takes in tuples generated by get_candidate_prons (one per line, separated
# by ";"), of the form:
# word;pron;base-word;base-pron;rule-name;de-stress[;rule-score]
# [note: we mean that the last element, the numeric score of the rule, is optional]
# and it outputs a (generally shorter) list
# of the same form.


# For each word:
  # For each (base-word,base-pron):
  #  Eliminate "more-general" rules as follows:
  #    For each pair of rules applying to this (base-word, base-pron):
  #      If pair is in more-general hash, disallow more general one.
  # Let the output be: for each (base-word, base-pron, rule):
  # for (destress-prefix) in [yes, no], do:
  #  print out the word input, the rule-name, [destressed:yes|no], and the new pron.


if (@ARGV != 1 && @ARGV != 2) {
  die "Usage: limit_candidate_prons.pl rule_hierarchy [candidate_prons] > limited_candidate_prons";
}

$hierarchy = shift @ARGV;
open(H, "<$hierarchy") || die "Opening rule hierarchy $hierarchy";

while(<H>) {
  chop;
  m:.+;.+: || die "Bad rule-hierarchy line $_";
  $hierarchy{$_} = 1; # Format is: if $rule1 is the string form of the more specific rule
  # and $rule21 is that string form of the more general rule, then $hierarchy{$rule1.";".$rule2}
  # is defined, else undefined.
}


sub process_word;

undef $cur_word;
@cur_lines = ();

while(<>) {
  # input, output is:
  # word;pron;base-word;base-pron;rule-name;destress;score
  chop;
  m:^([^;]+);: || die "Unexpected input: $_";
  $word = $1;
  if (!defined $cur_word || $word eq $cur_word) {
    if (!defined $cur_word) { $cur_word = $word; }
    push @cur_lines, $_;
  } else {
    process_word(@cur_lines); # Process a series of suggested prons
    # for a particular word.
    $cur_word = $word;
    @cur_lines = ( $_ ); 
  }
}
process_word(@cur_lines);
  
sub process_word {
  my %pair2rule_list; # hash from $baseword.";".$baseword to ref
  # to array of [ line1, line2, ... ].
  my @cur_lines = @_;
  foreach my $line (@cur_lines) {
    my ($word, $pron, $baseword, $basepron, $rulename, $destress, $rule_score) = split(";", $line);
    my $key = $baseword.";".$basepron;
    if (defined $pair2rule_list{$key}) {
      push @{$pair2rule_list{$key}}, $line; # @{...} derefs the array pointed to 
      # by the array ref inside {}. 
    } else {
      $pair2rule_list{$key} = [ $line ]; # [ $x ] is new anonymous array with 1 elem ($x)
    }
  }
  while ( my ($key, $value) = each(%pair2rule_list) ) {
    my @lines = @$value; # array of lines that are for this (baseword,basepron).
    my @stress, @rules; # Arrays of stress markers and rule names, indexed by
    # same index that indexes @lines.
    for (my $n = 0; $n < @lines; $n++) {
      my $line = $lines[$n];
      my ($word, $pron, $baseword, $basepron, $rulename, $destress, $rule_score) = split(";", $line);
      $stress[$n] = $destress;
      $rules[$n] = $rulename;
    }
    for (my $m = 0; $m < @lines; $m++) {
      my $ok = 1; # if stays 1, this line is OK.
      for (my $n = 0; $n < @lines; $n++) {
        if ($m != $n && $stress[$m] eq $stress[$n]) {
          if (defined $hierarchy{$rules[$n].";".$rules[$m]}) {
            # Note: this "hierarchy" variable is defined if $rules[$n] is a more
            # specific instances of $rules[$m], thus invalidating $rules[$m].
            $ok = 0;
            last; # no point iterating further.
          }
        }
      }
      if ($ok != 0) {
        print $lines[$m] . "\n";
      }
    }
  }
}
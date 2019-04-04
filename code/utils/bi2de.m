function d = bi2de(b, varargin)
%BI2DE Convert binary vectors to decimal numbers.
%   D = BI2DE(B) converts a binary vector B to a decimal value D. When B is
%   a matrix, the conversion is performed row-wise and the output D is a
%   column vector of decimal values. The default orientation of the binary
%   input is Right-MSB; the first element in B represents the least
%   significant bit.
%
%   In addition to the input matrix, two optional parameters can be given:
%
%   D = BI2DE(...,P) converts a base P vector to a decimal value.
%
%   D = BI2DE(...,MSBFLAG) uses MSBFLAG to determine the input orientation.
%   MSBFLAG has two possible values, 'right-msb' and 'left-msb'.  Giving a
%   'right-msb' MSBFLAG does not change the function's default behavior.
%   Giving a 'left-msb' MSBFLAG flips the input orientation such that the
%   MSB is on the left.
%
%   Examples:
%       B = [0 0 1 1; 1 0 1 0];
%       T = [0 1 1; 2 1 0];
%
%       D = bi2de(B)     
%       E = bi2de(B,'left-msb')     
%       F = bi2de(T,3)
%
%   See also DE2BI.

%   Copyright 1996-2015 The MathWorks, Inc.

  %%% Input validation:
  if nargin<1 || nargin>3
    error(message('comm:bi2de:IncorrectNumInputs'));
  end
  [b_double, inType, p] = validateInputs(b, varargin{:});

  %%% The binary to decimal conversion:
  max_length = 1024;
  pow2vector = p.^(0:1:(size(b_double,2)-1));
  size_B = min(max_length,size(b_double,2));
  d_double = b_double(:,1:size_B)*pow2vector(:,1:size_B).';

  % handle the Infs...
  d_double( max(b_double(:,max_length+1:size(b_double,2)).') == 1 ) = inf;

  %%% Output data type conversion:
  if ~any(strcmp(inType, {'double', 'logical'}))
      % If output fits, convert output to the input data type. Otherwise
      % convert to the smallest datatype that can fit the entire output
      % value, subject to the constraint that 'int' inputs do not output
      % 'uint', and vice versa.
      d = convertToNextSmallestContainingType(d_double, inType);
  else
      d = d_double;
  end
end

function [b, inType, p] = validateInputs(b, varargin)

  %%% 1st argument validation:
  validateattributes(b, {'logical', 'numeric'}, {'nonempty', 'real', ...
    'integer', 'nonnegative' }, '', 'binary vector b', 1);
  
  inType = class(b);
  b = double(b);  % All internal processing is done in double representation

  %%% Secondary input arguments:
  p = 2; % default base

  for i=1:length(varargin)

    thisArg = varargin{i};
    %%% Base argument (p):
    
    if isnumeric(thisArg)

      validateattributes(thisArg, {'numeric'}, {'nonempty', 'scalar', ...
                    'real', 'integer', 'finite', '>', 1}, '', 'base p', 1+i);
      
      p	= thisArg; % Set up the base to convert from.

    %%% Flag argument 'left-msb' / 'right-msb':
    elseif ischar(thisArg) 

      if ~any(strcmp(thisArg, {'left-msb', 'right-msb'}))
        error(message('comm:bi2de:InvalidMSBFlag'));
      end

      if strcmp(thisArg, 'left-msb')
        b = fliplr(b);
      end
    else
      error(message('comm:bi2de:InvalidInputArg'));
    end
  end

  if max(max(b(:))) > (p-1)
    error(message('comm:bi2de:InvalidInputElement'));
  end
end

function d = convertToNextSmallestContainingType(d_double, inType)

    if strncmp(inType, 'int', 3) % keep signed attribute
      
      maxValueTable   ={ 'int8',   double(intmax('int8'));
                         'int16',  double(intmax('int16'));
                         'int32',  double(intmax('int32'));
                         'int64',  double(intmax('int64')) };

    else                        % keep unsigned attribute
      maxValueTable   ={ 'uint8',   double(intmax('uint8'));
                         'uint16',  double(intmax('uint16'));
                         'uint32',  double(intmax('uint32'));
                         'uint64',  double(intmax('uint64')) };
    end
    maxValueTable = [maxValueTable; ...
                          {'single', realmax('single');
                           'double', realmax('double')} ];
                           
    inputIdx = find(strcmp(inType, maxValueTable(:, 1) )); % should not go below this offset
    % the maximum of the outputs determines the data type of the output array:
    findIdx = find( [maxValueTable{inputIdx:end, 2}] >= max(d_double), 1, 'first');
    smallestContainingType = maxValueTable{ inputIdx-1 + findIdx,   1};

    d = cast(d_double, smallestContainingType);
end